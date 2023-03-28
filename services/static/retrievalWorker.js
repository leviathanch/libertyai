function getBotParagraphs(count, hash, wholeString, resolve, reject) {
    var http = new XMLHttpRequest();
    http.open("GET", "/chatbot/get_part?id="+hash, true);
    http.setRequestHeader("Content-type", "application/json; charset=utf-8");
    http.onload = function () {
        var text = this.responseText
        wholeString += text
        self.postMessage({count: count, text: text});
        if ( wholeString.search("[DONE]") == -1 ) {
            getBotParagraphs(count+1, hash, wholeString);
        }
    };
    http.send();
};

let retrievalPromise = (hash) => {
    return new Promise((resolve, reject) => {
        var wholeString = "";
        getBotParagraphs(0, hash, wholeString, resolve, reject)
    })
}

self.onmessage = (event) => {
    retrievalPromise(event.data['hash']).then(function () {
        this.close();
    });
};
