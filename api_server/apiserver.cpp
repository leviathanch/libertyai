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

#include <bits/stdc++.h>

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
std::mutex emb_mtx;

// all the running threads (uuid: thread):
std::map<std::string, std::thread*> generator_threads;

// tokens generated for the treads (uuid: list)
std::map<std::string, std::vector<std::string>> available_tokens;

std::vector<std::string> last_partial_stops;
std::vector<std::string> last_partial_special;

struct liberty_args {
    std::string model;
    std::vector<std::string> stop;
    std::unordered_map<llama_token, float> logit_bias;
    int n_threads;
    int n_batch;
    float repeat_penalty;
    int32_t repeat_last_n;
    float temp;
    float top_p;
    int32_t top_k;
    int32_t n_ctx;
    bool input_noecho;
    int verbosity_level;
    bool embedding;
    float tfs_z;
    bool penalize_nl;
    int mirostat;
    float typical_p;
    float mirostat_eta;
    float mirostat_tau;
    float frequency_penalty;
    float presence_penalty;
};

bool contains_stop(std::string text, std::vector<std::string> tokens) {
    for(auto token: tokens) {
        if (text.size() < token.size()) {
            continue;
        }
        if( text.find(token) != std::string::npos ) {
            return true;
        }
    }
    return false;
}

bool is_partial_stop(std::string text, std::vector<std::string> tokens) {
    for(auto token: tokens) {
        if (token.size() < text.size()) {
            continue;
        }
        if( token.find(text) != std::string::npos ) {
            return true;
        }
    }
    return false;
}

bool contains_special(std::string text) {
    for(int i=0; i<text.size(); i++) {
        if( ( text[i] & 0xffffff00 ) == 0xffffff00 ) {
            return true;
        }
    }
    return false;
}

int predict_text(
        llama_context *ctx,
        liberty_args params,
        std::string uuid,
        std::string prompt
    )
{
    std::stringstream generated_text;

    bool input_noecho  = params.input_noecho;
    int n_past     = 0;
    int n_consumed = 0;

    const int n_ctx = llama_n_ctx(ctx);
    // tokenize the prompt
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx, prompt, true);
    if ((int) embd_inp.size() > n_ctx - 4) {
        available_tokens[uuid].push_back("[DONE]");
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        std::cout << "Prompt:\n" << prompt << std::flush;
        return 0;
    }
    int n_keep = (int)embd_inp.size();
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
                break;
            }
        }

        n_past += embd.size();
        embd.clear();

        if ((int) embd_inp.size() <= n_consumed) {
           // out of user input, sample next token
            const float   temp            = params.temp;
            const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
            const float   top_p           = params.top_p;
            const float   tfs_z           = params.tfs_z;
            const float   typical_p       = params.typical_p;
            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = params.repeat_penalty;
            const float   alpha_presence  = params.presence_penalty;
            const float   alpha_frequency = params.frequency_penalty;
            const int     mirostat        = params.mirostat;
            const float   mirostat_tau    = params.mirostat_tau;
            const float   mirostat_eta    = params.mirostat_eta;
            const bool    penalize_nl     = params.penalize_nl;

            // optionally save the session on first sample (for faster prompt loading next time)
            /*if (!path_session.empty() && need_to_save_session) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            }*/

            llama_token id = 0;
            auto logits = llama_get_logits(ctx);
            auto n_vocab = llama_n_vocab(ctx);

            // Apply params.logit_bias map
            for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                logits[it->first] += it->second;
            }

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // Apply penalties
            float nl_logit = logits[llama_token_nl()];
            auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
            llama_sample_repetition_penalty(ctx, &candidates_p,
                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                last_n_repeat, repeat_penalty);
            llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                last_n_repeat, alpha_frequency, alpha_presence);
            if (!penalize_nl) {
                logits[llama_token_nl()] = nl_logit;
            }

            if (temp <= 0) {
                // Greedy sampling
                id = llama_sample_token_greedy(ctx, &candidates_p);
            } else {
                if (mirostat == 1) {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    const int mirostat_m = 100;
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                } else if (mirostat == 2) {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                } else {
                    // Temperature sampling
                    llama_sample_top_k(ctx, &candidates_p, top_k);
                    llama_sample_tail_free(ctx, &candidates_p, tfs_z);
                    llama_sample_typical(ctx, &candidates_p, typical_p);
                    llama_sample_top_p(ctx, &candidates_p, top_p);
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    id = llama_sample_token(ctx, &candidates_p);
                }
            }
            // printf("`%d`", candidates_p.size);
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
                    break;
                }
            }
        }
        if ( !input_noecho || params.verbosity_level > 0 ) {
            for (auto id : embd) {
                std::string substr = llama_token_to_str(ctx, id);
                if ( !input_noecho ) {
                    generated_text << substr;
                    if(contains_special(substr)) {
                        last_partial_special.push_back(substr);
                    } else {
                        std::stringstream tmptok;
                        if(last_partial_special.size()) {
                            for(auto st: last_partial_special) {
                                tmptok << st;
                            }
                            last_partial_special.clear();
                            available_tokens[uuid].push_back(tmptok.str());
                        }
                        if(is_partial_stop(substr, params.stop)) {
                            last_partial_stops.push_back(substr);
                        } else {
                            std::stringstream tmptok;
                            for(auto st: last_partial_stops) {
                                tmptok << st;
                            }
                            if ( ! contains_stop(tmptok.str(), params.stop) ) {
                                for(auto st: last_partial_stops) {
                                    available_tokens[uuid].push_back(st);
                                }
                                available_tokens[uuid].push_back(substr);
                                last_partial_stops.clear();
                            }
                        }
                    }
                }
                if ( params.verbosity_level > 0 ) {
                    std::cout << substr << std::flush;
                }
            }
            if(contains_stop(generated_text.str(), params.stop)) {
                last_partial_stops.clear();
                break;
            }
        }
        // end of text token
        if ( !embd.empty() && embd.back() == llama_token_eos() ) {
            break;
        }
        mtx.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    mtx.unlock();
    available_tokens[uuid].push_back("[DONE]");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return 0;
}

void generate_embedding(
        llama_context *ctx,
        liberty_args params,
        std::string prompt,
        StringBuffer& response_buffer
    )
{
    emb_mtx.lock();
    Document response;
    response.SetObject();
    Value message_value;
    message_value.SetArray();

    int n_past = 0;
    auto embd_inp = ::llama_tokenize(ctx, prompt, true);
    if (embd_inp.size() > 0) {
        if (llama_eval(ctx, embd_inp.data(), embd_inp.size(), n_past, params.n_threads)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            response.AddMember("embedding", message_value, response.GetAllocator());
            Writer<StringBuffer> writer(response_buffer);
            response.Accept(writer);
            return;
        }
    }
    int n_embd = llama_n_embd(ctx);
    auto embeddings = llama_get_embeddings(ctx);
    auto& allocator = response.GetAllocator();
    Value jsonvalue;
    if( n_embd > 0 ) {
        for (int i = 0; i < n_embd; i++) {
            jsonvalue.SetFloat(embeddings[i]);
            message_value.PushBack(Document(&allocator).CopyFrom(jsonvalue, allocator), allocator);
        }
    }
    emb_mtx.unlock();

    response.AddMember("embedding", message_value, response.GetAllocator());
    Writer<StringBuffer> writer(response_buffer);
    response.Accept(writer);
}

void deploy_generation(
        llama_context *ctx,
        liberty_args params,
        std::string prompt,
        StringBuffer& response_buffer
    )
{
    boost::uuids::uuid uuid;
    std::stringstream uuid_str;
    Document response;
    response.SetObject();
    Value message_value;
    uuid = boost::uuids::random_generator()();
    uuid_str << uuid;
    generator_threads[uuid_str.str()] = new std::thread(predict_text, ctx, params, uuid_str.str(), prompt);
    message_value.SetString(uuid_str.str().c_str(), response.GetAllocator());
    response.AddMember("uuid", message_value, response.GetAllocator());
    Writer<StringBuffer> writer(response_buffer);
    response.Accept(writer);
}

void return_done(
        StringBuffer& response_buffer
    )
{
    Document response;
    response.SetObject();
    Value message_value;
    message_value.SetString("[DONE]");
    response.AddMember("text", message_value, response.GetAllocator());
    Writer<StringBuffer> writer(response_buffer);
    response.Accept(writer);
}

void fetch_tokens(
        std::string uuid,
        std::string index,
        StringBuffer& response_buffer
    )
{
    Document response;
    response.SetObject();
    Value message_value;
    int i = std::stoi(index);
    if( !available_tokens.count(uuid) && !generator_threads.count(uuid) ) {
        message_value.SetString("[DONE]");
        response.AddMember("text", message_value, response.GetAllocator());
        Writer<StringBuffer> writer(response_buffer);
        response.Accept(writer);
    } else {

        while ( available_tokens[uuid].size() < i+1 ) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        std::string text = available_tokens[uuid][i];
        message_value.SetString(text.c_str(), response.GetAllocator());

        // clean up. we're done here
        if ( text == "[DONE]" ) {
            generator_threads[uuid]->join();
            delete generator_threads[uuid];
            generator_threads.erase(uuid);
            available_tokens.erase(uuid);
        }

        response.AddMember("text", message_value, response.GetAllocator());
        Writer<StringBuffer> writer(response_buffer);
        response.Accept(writer);
    }
}

void handle_request(
        llama_context *ctx,
        llama_context *emb_ctx,
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
            liberty_args new_params = params;
            new_params.embedding = false;
            new_params.top_k = (doc.HasMember("top_k") && doc["top_k"].IsString()) ? std::stoi(doc["top_k"].GetString()) : params.top_k;
            new_params.top_p = (doc.HasMember("top_p") && doc["top_p"].IsString()) ? std::stof(doc["top_p"].GetString()) : params.top_p;
            new_params.repeat_last_n = (doc.HasMember("repeat_last_n") && doc["repeat_last_n"].IsString()) ? std::stoi(doc["repeat_last_n"].GetString()) : params.repeat_last_n;
            new_params.temp = (doc.HasMember("temp") && doc["temp"].IsString()) ? std::stof(doc["temp"].GetString()) : params.temp;
            new_params.repeat_penalty = (doc.HasMember("repeat_penalty") && doc["repeat_penalty"].IsString()) ? std::stof(doc["repeat_penalty"].GetString()) : params.repeat_penalty;
            if (doc.HasMember("stop") && doc["stop"].IsArray()) {
                for (auto& v : doc["stop"].GetArray()) {
                    if(v.IsString()) {
                        new_params.stop.push_back(v.GetString());
                    }
                }
            }
            if (doc.HasMember("echo") && doc["echo"].IsString()) {
                std::string have_echo = doc["echo"].GetString();
                if(have_echo=="yes") {
                    new_params.input_noecho = false;
                }
                if(have_echo=="no") {
                    new_params.input_noecho = true;
                }
            }
            new_params.n_batch = params.n_batch;
            std::string text = doc["text"].GetString();
            deploy_generation(ctx, new_params, text, response_buffer);
        } else return;
    } else if (request.method() == boost::beast::http::verb::post && request.target() == "/api/completion/fetch") {
        if (doc.HasMember("uuid") && doc["uuid"].IsString()&&doc.HasMember("index") && doc["index"].IsString()) {
            std::string uuid = doc["uuid"].GetString();
            std::string index= doc["index"].GetString();
            fetch_tokens(uuid, index, response_buffer);
        } else {
            return_done(response_buffer);
        }
    } else if (request.method() == boost::beast::http::verb::post && request.target() == "/api/embedding") {
        if (doc.HasMember("text") && doc["text"].IsString()) {
            liberty_args new_params;
            new_params.n_threads = params.n_threads;
            new_params.input_noecho = params.input_noecho;
            new_params.model = params.model;
            new_params.n_batch = params.n_batch;
            new_params.embedding = true;
            std::string text = doc["text"].GetString();
            generate_embedding(emb_ctx, new_params, text, response_buffer);
        }
    } else {
        std::cerr << "Invalid request: " << request.method_string() << " " << request.target() << std::endl;
        return_done(response_buffer);
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
    lparams.embedding  = false;
    ctx = llama_init_from_file(params.model.c_str(), lparams);
    return ctx;
}

llama_context *load_embedding(liberty_args& params) {
    llama_context * ctx;
    auto lparams = llama_context_default_params();
    lparams.n_ctx      = params.n_ctx;
    lparams.n_parts    = 1;
    lparams.seed       = time(NULL);
    lparams.f16_kv     = 2;
    lparams.use_mlock  = 0;
    lparams.embedding  = true;
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
    params.n_ctx = 2048;
    params.input_noecho = true;
    params.verbosity_level = 0;
    params.tfs_z = 1.00f;
    params.penalize_nl = false;
    params.typical_p = 1.00f;
    params.frequency_penalty = 0.00f;
    params.presence_penalty = 0.00f;
    params.mirostat = 0;
    params.mirostat_tau = 5.00f;
    params.mirostat_eta = 0.10f;

    po::options_description desc("Options for the API server");
    desc.add_options()
        ("help,h", "Show this help message")
        ("model,m", po::value<std::string>(&params.model), "Path to the model")
        ("threads,t", po::value<int>(&params.n_threads), "Number of threads")
        ("context,N", po::value<int>(&params.n_ctx), "Context size")
        ("batch,B", po::value<int>(&params.n_batch), "Batch size")
        ("verbose,v", po::value<int>()->implicit_value(1), "enable verbosity (optionally specify level)")
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
    if (vm.count("verbose")) {
        params.verbosity_level = vm["verbose"].as<int>();
    }

    return true;
}

int main(int argc, char ** argv) {
    llama_context *ctx;
    llama_context *emb_ctx;
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

    // load the model
    emb_ctx = load_embedding(params);
    if (emb_ctx == NULL) {
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
        handle_request(ctx, emb_ctx, params, socket, response_buffer);

        // Send the HTTP response back to the client
        std::string response_string = "HTTP/1.1 200 OK\r\n";
        response_string += "Content-Type: application/json\r\n";
        response_string += "Content-Length: " + std::to_string(response_buffer.GetSize()) + "\r\n";
        response_string += "\r\n";
        response_string += response_buffer.GetString();
        response_string += "\n";
        boost::asio::write(socket, boost::asio::buffer(response_string));

        // Close the connection
        try {
            socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
            socket.close();
        }
        catch (...) {
            std::cout << "Connection not properly closed" << std::endl;
        }
    }
    return 0;
}
