function getChatWorkObject() {
    var div = document.createElement("div");
    div.className = "min-h-[20px] flex flex-col items-start gap-4 whitespace-pre-wrap";
    return div
}

function getChatTextDiv(workDiv) {
    var div0 = document.createElement("div");
    var div1 = document.createElement("div");
    var div2 = document.createElement("div");
    var div3 = document.createElement("div");
    div0.className = "relative flex w-[calc(100%-50px)] flex-col gap-1 md:gap-3 lg:w-[calc(100%-115px)]";
    div1.className = "flex flex-grow flex-col gap-3";
    div3.className = "flex justify-between";
    div1.append(workDiv)
    div0.append(div1)
    div0.append(div3)
    return div0
}

function getAvatarDiv(avatar, username) {
    var div0 = document.createElement("div");
    var div1 = document.createElement("div");
    var namediv = document.createElement("div");
    var img = document.createElement("img");
    img.height = img.width = 40;
    img.src = avatar;
    div0.className = "w-[60px] flex flex-col relative items-end";
    div1.className = "relative h-[40px] w-[40px] p-1 rounded-sm text-white flex items-center justify-center";
    namediv.innerHTML = username;
    namediv.className = "avatarName justify-center w-[60px] p-1 rounded-sm text-white flex items-center justify-center";
    div1.append(img);
    div0.append(div1);
    div0.append(namediv);
    return div0
}

function getChatDiv(text, color, avatar, username) {
    var top = document.createElement("div");
    var div0 = document.createElement("div");
    var div1 = document.createElement("div");
    div0.className = "flex flex-col items-center text-sm dark:bg-gray-800 "+color;
    div1.className = "text-base gap-4 md:gap-6 md:max-w-2xl lg:max-w-2xl xl:max-w-3xl p-4 md:py-6 flex lg:px-0 m-auto";
    div1.append(getAvatarDiv(avatar, username))
    div1.append(getChatTextDiv(text))
    div0.append(div1)
    top.className = "group w-full text-gray-800 dark:text-gray-100 border-b border-black/10 dark:border-gray-900/50 "+color;
    top.append(div1)
    return top
}

var queues = {};

function addToTypeWriterQueue(hash, data) {
    queues[hash].push(data);
}

function startTypeWriterJob(hash, workerObject) {
    
    const typeWriterPromise = function (o, i, d) {
        return new Promise(
            function (resolve, reject) {
                o.innerText += d.charAt(i);
                setTimeout(resolve, 100);
            }
       );
    };

    const recursiveTypeWriterPromise = function (o, i, d) {
        return new Promise(
            function (resolve, reject) {
                if ( i < d.length ) {
                    typeWriterPromise(o, i, d).then(
                        function () {
                            recursiveTypeWriterPromise(o, i+1, d).then(
                                function () {
                                    resolve();
                                }
                            );
                        }
                    )
                } else {
                    resolve();
                }
            }
        );
    };

    const QueuePromise = function (hash, o) {
        return new Promise(
            function (resolve, reject) {
                if ( queues[hash].length > 0 ) {
                    m = queues[hash].shift();
                    if (m.data === "[DONE]") {
                        resolve("done");
                    } else {
                        recursiveTypeWriterPromise(o, 0, m.data).then(
                            function () {
                                resolve("idle");
                            }
                        );
                    }
                } else {
                    resolve("idle");
                }
            }
       );
    };

    const recursiveQueueJob = (hash, workerObject) => {
        QueuePromise(hash, workerObject).then(function (state) {
            if ( state === "done") {
                workerObject.classList.remove("blinkyChat");
                workerObject.classList.add("normalChat");
            } else {
                setTimeout(recursiveQueueJob, 100, hash, workerObject);
            }
        });
    }

    queues[hash]=[]
    recursiveQueueJob(hash, workerObject);
};

function getBotResponse() {
    if ($("#textInput").val()==="") {
        return 0;
    }
    var rawText = $("#textInput").val();
    $("#textInput").val("");
    var chatWorkObject = getChatWorkObject();
    var newField = getChatDiv(
        chatWorkObject,
        "dark:bg-gray-800",
        get_human_avatar(),
        get_user_name()
    );
    $("#chatbox").append(newField);
    chatWorkObject.innerHTML = rawText;
    var chatBotWorkObject = getChatWorkObject();
    var newChatBotField = getChatDiv(
        chatBotWorkObject,
        "bg-gray-50 dark:bg-[#444654]",
        get_liberty_avatar(),
        "LibertyAI"
    );
    $("#chatbox").append(newChatBotField);
    var txt = document.createElement("div");
    txt.classList.add("blinkyChat");
    chatBotWorkObject.append(txt);
    $.get("/chatbot/start_generation", { msg: rawText }).done( function(uuid) {
        var eventSource = new EventSource("/chatbot/stream?uuid="+uuid);
        eventSource.onmessage = function(e) {
            console.log(e.data);
            if ( e.data === "[DONE]") {
                this.close();
                txt.classList.replace("blinkyChat", "normalChat");
            } else if ( e.data !== "[BUSY]") {
                addToTypeWriterQueue(uuid, e);
            }
        };
        startTypeWriterJob(uuid, txt);
    });
}

document.querySelector('emoji-picker').addEventListener('emoji-click', e => {
    var cursorPos = $('#textInput').prop('selectionStart');
    var v = $('#textInput').val();
    var textBefore = v.substring(0,  cursorPos);
    var textAfter  = v.substring(cursorPos, v.length);
    $('#textInput').val(textBefore + e.detail.unicode + textAfter);
});

document.getElementById('textInput').addEventListener('keydown', function (e) {
    const keyCode = e.which || e.keyCode;
    if (keyCode === 13 && !e.shiftKey) {
        e.preventDefault();
        getBotResponse();
        this.style.height = 0;
        this.style.height = (this.scrollHeight) + "px";
    }
});

document.getElementById('textInput').addEventListener('click', function() {
    document.getElementById("emojiPicker").style.display = "none";
});

document.getElementById('buttonEmoji').addEventListener('click', function() {
    var visible = document.getElementById("emojiPicker").style.display;
    if ( visible === "none" ) {
        visible = "block";
    } else if ( visible === "block" ) {
        visible = "none";
    }
    document.getElementById("emojiPicker").style.display = visible;
});

document.getElementById('textInput').addEventListener('input', function () {
    this.style.height = 0;
    this.style.height = (this.scrollHeight) + "px";
    document.getElementById("emojiPicker").style.display = "none";
});

document.getElementById('buttonInput').addEventListener('click', function() {
    getBotResponse();
});

window.onload = function(e){
$.get("/chatbot/get_chat_history").done( function(history) {
        for (let i = 0; i < history.length; i++) {
            if (history[i]['Human']) {
                var chatWorkObject = getChatWorkObject();
                var newField = getChatDiv(
                    chatWorkObject,
                    "dark:bg-gray-800",
                    get_human_avatar(),
                    get_user_name()
                );
                $("#chatbox").append(newField);
                chatWorkObject.innerHTML = history[i]['Human'];
            }
            if (history[i]['LibertyAI']) {
                var chatWorkObject = getChatWorkObject();
                var newField = getChatDiv(
                    chatWorkObject,
                    "bg-gray-50 dark:bg-[#444654]",
                    get_liberty_avatar(),
                    "LibertyAI"
                );
                $("#chatbox").append(newField);
                chatWorkObject.innerHTML = history[i]['LibertyAI']
            }
        }
    });
}
