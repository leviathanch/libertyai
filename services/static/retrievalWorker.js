const retrievalPromise = function (hash) {
    return new Promise(
        function (resolve, reject) {
            var http = new XMLHttpRequest();
            http.open("GET", "/chatbot/get_part?id="+hash, true);
            http.setRequestHeader("Content-type", "application/json; charset=utf-8");
            http.onload = function () {
                var text = this.responseText
                self.postMessage(text);
                if ( text !== "[DONE]") {
                    retrievalPromise(hash).then(function () {
                        setTimeout(resolve, 50, "iterating");
                    });
                } else {
                    setTimeout(resolve, 50, "done");
                }
            };
            http.send();
        }
    );
};

self.onmessage = (event) => {
    retrievalPromise(event.data).then(function (state) {
        this.close();
    });
};
