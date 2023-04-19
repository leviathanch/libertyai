const retrievalPromise = function (uuid, index) {
    return new Promise(
        function (resolve, reject) {
            var http = new XMLHttpRequest();
            http.open("GET", "/chatbot/get_part?uuid="+uuid+"&index="+index, true);
            http.setRequestHeader("Content-type", "application/json; charset=utf-8");
            http.onload = function () {
                var text = this.responseText
                console.log(text);
                if ( text === "[DONE]") {
                    self.postMessage(text);
                    resolve("done");
                } else if ( text === "[BUSY]") {
                    retrievalPromise(uuid, index ).then(function (state) {
                        setTimeout(resolve, 1000, "iterating");
                    });
                } else {
                    self.postMessage(text);
                    retrievalPromise(uuid, index+1 ).then(function (state) {
                        setTimeout(resolve, 200, "iterating");
                    });
                }
            };
            http.send();
        }
    );
};

self.onmessage = (event) => {
    retrievalPromise(event.data, 0).then(function (state) {
        this.close();
    });
};
