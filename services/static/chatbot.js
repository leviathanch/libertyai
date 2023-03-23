function getChatTextDiv(text) {
    var div0 = document.createElement("div");
    var div1 = document.createElement("div");
    var div2 = document.createElement("div");
    var div3 = document.createElement("div");
    div0.className = "relative flex w-[calc(100%-50px)] flex-col gap-1 md:gap-3 lg:w-[calc(100%-115px)]";
    div1.className = "flex flex-grow flex-col gap-3";
    div2.className = "min-h-[20px] flex flex-col items-start gap-4 whitespace-pre-wrap"
    div3.className = "flex justify-between"
    div2.append(text)
    div1.append(div2)
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
