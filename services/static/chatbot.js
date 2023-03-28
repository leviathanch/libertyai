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

var queue = [];
function addToTypeWriterQueue(data) {
    queue.push(data);
    queue = queue.sort((a,b) => a.count < b.count);
}

var currentWorkObject;

function setCurrentWorkObject(o) {
    currentWorkObject = o;
}

function typeWriter(o, i, data, resolve, reject) {
    if (i < data.text.length) {
        o.innerHTML += data.text.charAt(i);
        if(o.innerHTML.search("[[DONE]]")!=-1) {
            o.innerHTML = o.innerHTML.replace("[[DONE]]","");
        } else {
            setTimeout(typeWriter, 100, o, i+1, data);
        }
    }
};


function startTypeWriterJob() {
    var busy = false;

    const typeWriterPromise = (o, i, data) => {
        return new Promise( (resolve, reject) => {
            if (i < data.text.length) {
                o.innerHTML += data.text.charAt(i);
                if(o.innerHTML.search("[[DONE]]")!=-1) {
                    o.innerHTML = o.innerHTML.replace("[[DONE]]","");
                }
            }
            setTimeout(resolve, 50);
        }).then(function () {
            if (i < data.text.length) {
                setTimeout(typeWriterPromise, 100, o, i+1, data);
            }
        })
    }

    const recursiveTypeWriterJob = () => {
        return new Promise( (resolve, reject) => {
            if (busy) {
                resolve();
                return 0;
            }
            if ( queue.length > 0 && currentWorkObject ) {
                busy = true;
                data = queue.shift();
                console.log(data.text);
                typeWriterPromise(currentWorkObject, 0, data).then(function () {
                    busy = false;
                    setTimeout(resolve, 500);
                });
            }
            setTimeout(resolve, 500);
        }).then(function () {
            setTimeout(recursiveTypeWriterJob, 500);
        })
    }
    recursiveTypeWriterJob();
};




