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
                o.appendChild(
                    document.createTextNode(
                        d.charAt(i)
                    )
                );
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
                setTimeout(recursiveQueueJob, 500, hash, workerObject);
            }
        });
    }

    queues[hash]=[]
    recursiveQueueJob(hash, workerObject);
};
