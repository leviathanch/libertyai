#include "common.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <map>
#include <mutex>
#include <chrono>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <boost/program_options.hpp>

using namespace rapidjson;
namespace po = boost::program_options;

std::mutex mtx;

// all the running threads (uuid: thread):
std::map<std::string, std::thread*> generator_threads;

// tokens generated for the treads (uuid: list)
std::map<std::string, std::vector<std::string>> available_tokens;

struct liberty_args {
    std::string model;
    std::vector<std::string> antiprompt;
    int n_threads;
    int n_batch;
    bool ignore_eos;
    float repeat_penalty;
    int32_t repeat_last_n;
    float temp;
    float top_p;
    int32_t top_k;
    int32_t n_ctx;
};

int predict_text(
        llama_context *ctx,
        liberty_args& params,
        const std::string& uuid,
        const std::string& prompt
    )
{
    bool is_interacting = false;
    bool is_antiprompt = false;
    bool input_noecho  = true;

    int n_past     = 0;
    int n_consumed = 0;

    const int n_ctx = llama_n_ctx(ctx);
    // tokenize the prompt
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx, prompt, true);
    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }
    int n_keep = (int)embd_inp.size();

    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    std::vector<llama_token> embd;
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    while (true) {
        mtx.lock();
        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - n_keep;
                n_past = n_keep;
                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
            }
            if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                mtx.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                break;
            }
        }

        n_past += embd.size();
        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // out of user input, sample next token
            llama_token id = 0;
            auto logits = llama_get_logits(ctx);
            if (params.ignore_eos) {
                logits[llama_token_eos()] = 0;
            }
            id = llama_sample_top_p_top_k(ctx,
                    last_n_tokens.data() + n_ctx - params.repeat_last_n,
                    params.repeat_last_n, params.top_k, params.top_p, params.temp, params.repeat_penalty);
            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(id);
            // add it to the context
            embd.push_back(id);
            // echo this to console
            input_noecho = false;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    mtx.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    break;
                }
            }
        }
        if (!input_noecho) {
            for (auto id : embd) {
                std::string tok = llama_token_to_str(ctx, id);
                std::cout << tok << std::endl;
                available_tokens[uuid].push_back(tok);
            }
        }
        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
            available_tokens[uuid].push_back("[DONE]");
            mtx.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            break;
        }
        mtx.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return 0;
}

void deploy_generation(
        llama_context *ctx,
        liberty_args& params,
        std::string& prompt,
        StringBuffer& response_buffer
    )
{
    boost::uuids::uuid uuid;
    std::stringstream uuid_str;
    Document response;
    response.SetObject();
    Value message_value;
    rapidjson::Document::AllocatorType &allocator = response.GetAllocator();
    uuid = boost::uuids::random_generator()();
    uuid_str << uuid;
    generator_threads[uuid_str.str()] = new std::thread(predict_text, ctx, std::ref(params), uuid_str.str(), prompt);
    message_value.SetString(uuid_str.str().c_str(), allocator);
    response.AddMember("uuid", message_value, response.GetAllocator());
    Writer<StringBuffer> writer(response_buffer);
    response.Accept(writer);
}

void fetch_tokens(
        const std::string& uuid,
        const std::string& index,
        StringBuffer& response_buffer
    )
{
    Document response;
    response.SetObject();
    Value message_value;
    int i = stoi(index);
    rapidjson::Document::AllocatorType &allocator = response.GetAllocator();

    if( available_tokens[uuid].size() > i ) {
        std::string text = available_tokens[uuid][i];
        message_value.SetString(text.c_str(), allocator);
    } else {
        message_value.SetString("[BUSY]");
    }

    response.AddMember("text", message_value, response.GetAllocator());
    Writer<StringBuffer> writer(response_buffer);
    response.Accept(writer);
}

void handle_request(
        llama_context *ctx,
        liberty_args& params,
        boost::asio::ip::tcp::socket& socket,
        StringBuffer& response_buffer
    ) 
{
    boost::beast::http::request<boost::beast::http::string_body> request;
    boost::beast::http::request_parser<boost::beast::http::string_body> parser;
    boost::beast::error_code error;
    boost::asio::streambuf request_buffer;
    boost::beast::http::read(socket, request_buffer, request, error);
    if (error) {
      std::cerr << "Failed to parse HTTP request: " << error.message() << std::endl;
      return;
    }
    std::string body_str(request.body().data(), request.body().size());
    rapidjson::Document doc;
    doc.Parse(body_str.c_str());
    // Process the request based on the requested path
    if (request.method() == boost::beast::http::verb::post && request.target() == "/api/completion/submit") {
        if (doc.HasMember("text") && doc["text"].IsString()) {
            liberty_args new_params;
            new_params.n_threads = params.n_threads;
            new_params.model = params.model;
            new_params.top_k = (doc.HasMember("top_k") && doc["top_k"].IsString()) ? std::stoi(doc["top_k"].GetString()) : params.top_k;
            new_params.repeat_last_n = (doc.HasMember("repeat_last_n") && doc["repeat_last_n"].IsString()) ? std::stoi(doc["repeat_last_n"].GetString()) : params.repeat_last_n;
            new_params.top_p = (doc.HasMember("top_p") && doc["top_p"].IsString()) ? std::stof(doc["top_p"].GetString()) : params.top_p;
            new_params.temp = (doc.HasMember("temp") && doc["temp"].IsString()) ? std::stof(doc["temp"].GetString()) : params.temp;
            new_params.repeat_penalty = (doc.HasMember("repeat_penalty") && doc["repeat_penalty"].IsString()) ? std::stof(doc["repeat_penalty"].GetString()) : params.repeat_penalty;
            new_params.ignore_eos = false;
            new_params.n_batch = params.n_batch;

            std::string text = doc["text"].GetString();
            deploy_generation(ctx, new_params, text, response_buffer);
        } else return;
    } else if (request.method() == boost::beast::http::verb::post && request.target() == "/api/completion/fetch") {
        if (doc.HasMember("uuid") && doc["uuid"].IsString()&&doc.HasMember("index") && doc["index"].IsString()) {
            std::string uuid = doc["uuid"].GetString();
            std::string index= doc["index"].GetString();
            fetch_tokens(uuid, index, response_buffer);
        } else return;
    } else {
        std::cerr << "Invalid request: " << request.method_string() << " " << request.target() << std::endl;
        return;
    }
}

llama_context *load_model(liberty_args& params) {
    llama_context * ctx;
    auto lparams = llama_context_default_params();
    lparams.n_ctx      = params.n_ctx;
    lparams.n_parts    = 1;
    lparams.seed       = time(NULL);
    lparams.f16_kv     = 2;
    lparams.use_mlock  = 0;
    ctx = llama_init_from_file(params.model.c_str(), lparams);
    return ctx;
}

void print_sysinfo(liberty_args& params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n", params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
}

bool parse_params(int ac, char ** av, liberty_args &params) {
    params.model = "";
    params.n_threads = 1;
    params.top_k = 40;
    params.top_p = 0.95;
    params.temp = 0.7;
    params.repeat_penalty = 1.0;
    params.repeat_last_n = 64;
    params.n_batch = 32;
    params.n_ctx = 512;

    po::options_description desc("Options for the API server");
    desc.add_options()
        ("help,h", "Show this help message")
        ("model,m", po::value<std::string>(&params.model), "Path to the model")
        ("threads,t", po::value<int>(&params.n_threads), "Number of threads")
        ("context,N", po::value<int>(&params.n_ctx), "Context size")
        ("batch,B", po::value<int>(&params.n_batch), "Batch size")
        ;
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(ac, av, desc), vm);
    }
    catch (...) {
        std::cout << desc << std::endl;
        return false;
    }
    po::notify(vm);

    if (vm.count("help") || !vm.count("model")) {
        std::cout << desc << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    llama_context *ctx;
    liberty_args params;

    if (parse_params(argc, argv, params) == false) {
        return 1;
    }

    if ( params.model == "" ) {
        return 1;
    }

    // load the model
    ctx = load_model(params);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\nUse -m for specifying your own model path!\n", __func__, params.model.c_str());
        return 1;
    }

    // print system information
    print_sysinfo(params);

    boost::asio::io_service io_service;
    boost::asio::ip::tcp::acceptor acceptor(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 8080));

    while (true) {
        // Wait for an incoming connection
        boost::asio::ip::tcp::socket socket(io_service);
        acceptor.accept(socket);

        // Read the incoming HTTP request
        boost::asio::streambuf request_buffer;

        // Handle the request and generate a response
        StringBuffer response_buffer;
        handle_request(ctx, params, socket, response_buffer);

        // Send the HTTP response back to the client
        std::string response_string = "HTTP/1.1 200 OK\r\n";
        response_string += "Content-Type: application/json\r\n";
        response_string += "Content-Length: " + std::to_string(response_buffer.GetSize()) + "\r\n";
        response_string += "\r\n";
        response_string += response_buffer.GetString();
        response_string += "\n";
        boost::asio::write(socket, boost::asio::buffer(response_string));

        // Close the connection
        socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
        socket.close();
    }
    return 0;
}
