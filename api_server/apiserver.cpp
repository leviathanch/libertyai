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

using namespace rapidjson;

std::mutex mtx;

// all the running threads (uuid: thread):
std::map<std::string, std::thread*> generator_threads;

// tokens generated for the treads (uuid: list)
std::map<std::string, std::vector<std::string>> available_tokens;

int predict_text(
        llama_context *ctx,
        gpt_params params,
        const std::string& uuid,
        const std::string& prompt
    )
{
    bool is_interacting = false;
    bool is_antiprompt = false;
    bool input_noecho  = false;

    int n_past     = 0;
    int n_remain   = params.n_predict;
    int n_consumed = 0;

    const int n_ctx = llama_n_ctx(ctx);
    // tokenize the prompt
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx, prompt, true);
    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }
    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    }

    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    std::vector<llama_token> embd;
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    while (n_remain) {
        mtx.lock();
        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;
                n_past = params.n_keep;
                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
            }
            
            if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                break;
            }
        }

        n_past += embd.size();
        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // out of user input, sample next token
            const int32_t top_k          = params.top_k;
            const float   top_p          = params.top_p;
            const float   temp           = params.temp;
            const float   repeat_penalty = params.repeat_penalty;

            llama_token id = 0;
            auto logits = llama_get_logits(ctx);
            if (params.ignore_eos) {
                logits[llama_token_eos()] = 0;
            }
            id = llama_sample_top_p_top_k(ctx,
                    last_n_tokens.data() + n_ctx - params.repeat_last_n,
                    params.repeat_last_n, top_k, top_p, temp, repeat_penalty);
            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(id);
            // replace end of text token with newline token when in interactive mode
            if (id == llama_token_eos() && params.interactive) {
                id = llama_token_newline.front();
                if (params.antiprompt.size() != 0) {
                    // tokenize and inject first reverse prompt
                    const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
                    embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                }
            }
            // add it to the context
            embd.push_back(id);
            // echo this to console
            input_noecho = false;
            // decrement remaining sampling budget
            --n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }
        if (!input_noecho) {
            for (auto id : embd) {
                available_tokens[uuid].push_back(llama_token_to_str(ctx, id));
            }
        }
        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (params.interactive && (int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt
            if (params.antiprompt.size()) {
                std::string last_output;
                for (auto id : last_n_tokens) {
                    last_output += llama_token_to_str(ctx, id);
                }
                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                for (std::string & antiprompt : params.antiprompt) {
                    if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                        is_interacting = true;
                        is_antiprompt = true;
                        break;
                    }
                }
            }

            if (n_past > 0 && is_interacting) {
                std::string buffer;
                if (!params.input_prefix.empty()) {
                    buffer += params.input_prefix;
                    printf("%s", buffer.c_str());
                }

                std::string line;
                bool another_line = true;
                do {
                    if (line.empty() || line.back() != '\\') {
                        another_line = false;
                    } else {
                        line.pop_back(); // Remove the continue character
                    }
                    buffer += line + '\n'; // Append the line to the result
                } while (another_line);

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    auto line_inp = ::llama_tokenize(ctx, buffer, false);
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    n_remain -= line_inp.size();
                }
                input_noecho = true; // do not echo this again
            }
            if (n_past > 0) {
                is_interacting = false;
            }
        }
        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
            available_tokens[uuid].push_back("[DONE]");
            break;
        }
        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
        mtx.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    mtx.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return 0;
}

void deploy_generation(
        llama_context *ctx,
        gpt_params params,
        const std::string& prompt,
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
    generator_threads[uuid_str.str()] = new std::thread(predict_text, ctx, params, uuid_str.str(), prompt);
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
        gpt_params params,
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
            std::string text = doc["text"].GetString();
            deploy_generation(ctx, params, text, response_buffer);
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

llama_context *load_model(gpt_params& params) {
    llama_context * ctx;
    auto lparams = llama_context_default_params();
    lparams.n_ctx      = params.n_ctx;
    lparams.n_parts    = params.n_parts;
    lparams.seed       = (params.seed <= 0) ? time(NULL) : params.seed;
    lparams.f16_kv     = params.memory_f16;
    lparams.use_mlock  = params.use_mlock;
    ctx = llama_init_from_file(params.model.c_str(), lparams);
    return ctx;
}

void print_sysinfo(gpt_params& params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n", params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
}

int main(int argc, char ** argv) {
    llama_context *ctx;
    gpt_params params;
    if ( params.model == "" ) {
        return 1;
    }

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
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
        //boost::asio::read_until(socket, request_buffer, "\r\n\r\n");

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
