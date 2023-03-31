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

function getAvatarDiv(avatar) {
    var div0 = document.createElement("div");
    var div1 = document.createElement("div");
    var img = document.createElement("img");
    img.height = img.width = 40;
    img.src = avatar;
    div0.className = "w-[40px] flex flex-col relative items-end";
    div1.className = "relative h-[40px] w-[40px] p-1 rounded-sm text-white flex items-center justify-center";
    div1.append(img)
    div0.append(div1)
    return div0
}

function getChatDiv(text, color, avatar) {
    var top = document.createElement("div");
    var div0 = document.createElement("div");
    var div1 = document.createElement("div");
    div0.className = "flex flex-col items-center text-sm dark:bg-gray-800 "+color;
    div1.className = "text-base gap-4 md:gap-6 md:max-w-2xl lg:max-w-2xl xl:max-w-3xl p-4 md:py-6 flex lg:px-0 m-auto";
    div1.append(getAvatarDiv(avatar))
    div1.append(getChatTextDiv(text))
    div0.append(div1)
    top.className = "group w-full text-gray-800 dark:text-gray-100 border-b border-black/10 dark:border-gray-900/50 "+color;
    top.append(div1)
    return top
}

var queues = {};

function addToTypeWriterQueue(hash, data) {
    queues[hash].queue.push(data);
}

function endTypeWriterJob(hash) {
    queues[hash].active = false;
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
                            recursiveTypeWriterPromise(o, i+1, d);
                            setTimeout(resolve, 500);
                        }
                    )
                } else {
                    setTimeout(resolve, 500);
                }
            }
        );
    };

    const QueuePromise = function (hash, o) {
        return new Promise(
            function (resolve, reject) {
                if ( queues[hash].queue.length > 0 ) {
                    m = queues[hash].queue.shift();
                    recursiveTypeWriterPromise(o, 0, m.data).then(
                        function () {
                            setTimeout(resolve, 500);
                        }
                    );
                } else {
                    setTimeout(resolve, 500);
                }
            }
       );
    };

    const recursiveQueueJob = (hash, workerObject) => {
        if ( queues[hash].queue.length > 0 || queues[hash].active ) {
            QueuePromise(hash, workerObject).then(function () {
                setTimeout(recursiveQueueJob, 500, hash, workerObject);
            });
        }
    }

    queues[hash]={
        'active': true,
        'queue': [],
    }
    recursiveQueueJob(hash, workerObject);
};
