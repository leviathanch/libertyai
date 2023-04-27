// Get it for Wallpaper Engine: https://steamcommunity.com/sharedfiles/filedetails/?id=1755131079&result=1

// Sorry for the Tags in the audio but finding good Royalty-free Music isnt easy

var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
w = ctx.canvas.width = window.innerWidth;
h = ctx.canvas.height = window.innerHeight;

window.onresize = function() {
    w = ctx.canvas.width = window.innerWidth;
    h = ctx.canvas.height = window.innerHeight;
};

nrt=0;
npt=0;
dots=[];
lines=[];

config = {
    circleRadius: 80,
    multiplier:   40,
    colorSpeed:   20,
    hueStart:     220,
    glow:         0
}

var AudioContext = window.AudioContext || window.webkitAudioContext;
var recording = false;
var audioBuffer = {};

(function() {
    if (!AudioContext) {
        console.log("No Audio");
        return;
    }
})();

function initializeAudio(uuid) {
    actx = new AudioContext();
    console.log("Loading Audio Buffer: "+uuid);
    var xmlHTTP = new XMLHttpRequest();
    xmlHTTP.open(
        "GET",
        "/voicechat/audio?uuid="+uuid,
        true
    );
    xmlHTTP.responseType = "arraybuffer";
    xmlHTTP.onload = function(e) {
        console.log("Decoding Audio File Data");
        actx.decodeAudioData(
            this.response,
            function(buffer) {
                console.log("Ready");
                audioBuffer[uuid] = buffer;
                // Run
                analyser = actx.createAnalyser();
                analyser.fftSize = 256;
                analyser.smoothingTimeConstant = 0.6;
                analyser.maxDecibels = 0;
                analyser.minDecibels = -100;
                gainNode = actx.createGain();
                gainNode.connect(analyser);
                analyser.connect(actx.destination);
                frequencyDataLen = analyser.frequencyBinCount;
                frequencyData = new Uint8Array(frequencyDataLen);
                play(uuid);
            },
            function() {
                console.log("Error decoding audio data");
            }
        );
    };
    xmlHTTP.send();
}

function emitDot(){
    if (dots.length > 150) { return; }
    dots.push({
        xp:  w/2,
        yp:  h/2,
        xv:  Math.random()*0.4-0.2,
        yv:  Math.random()*0.4-0.2,
        rad: Math.random()*(15-2)+2,
        hue: Math.random()*50-25
    });
}

function emitLine(){
    if (lines.length > 50) { return; }
    lines.push({
        xp:  w/2,
        yp:  h/2,
        xv:  Math.random()*0.4-0.2,
        yv:  Math.random()*0.4-0.2,
        hue: Math.random()*50-25
    });
}

function clear(){
    var avg = averageFrequency();
    ctx.beginPath();
    var grd = ctx.createLinearGradient(w/2, 0, w/2, h);
    grd.addColorStop(0, "hsl("+(config.hueStart + npt * config.colorSpeed)+", 35%, 10%");
    grd.addColorStop(1, "hsl("+(config.hueStart + npt * config.colorSpeed)+", 75%, 5%");
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, w, h);
    ctx.closePath();
}

function drawDots(){
    var avg = averageFrequency();
    
    for(i = 0; i < dots.length; i++){
        ctx.beginPath();
        var grd = ctx.createRadialGradient(dots[i].xp + dots[i].rad, dots[i].yp + dots[i].rad, 0, dots[i].xp + dots[i].rad, dots[i].yp + dots[i].rad, dots[i].rad);
        grd.addColorStop(0, "hsla("+(config.hueStart + npt * config.colorSpeed + dots[i].hue)+", 50%, 50%, "+(avg / 400)+"%)");
        grd.addColorStop(1, "hsla("+(config.hueStart + npt * config.colorSpeed + dots[i].hue)+", 50%, 50%, 0%)");
        ctx.fillStyle = grd;
        ctx.fillRect(dots[i].xp, dots[i].yp, dots[i].rad*2, dots[i].rad*2);
        ctx.closePath();
        
        if(dots[i].xp > w || dots[i].xp < 0 || dots[i].yp > w || dots[i].yp < 0){
            dots[i] = dots[dots.length-1];
            dots.pop();
        } else {
            dots[i].xp += dots[i].xv * Math.pow(avg/1000, 1.5);
            dots[i].yp += dots[i].yv * Math.pow(avg/1000, 1.5);
        }
    }
}

function drawLines() {
    var avg = averageFrequency();
    var maxDist = 150;

    for (var i = 0; i < lines.length; i++) {
        for (var j = 0; j < lines.length; j++) {
            var proDist = 100 / maxDist;
            var opacity = 100 - dist(lines[j].xp, lines[j].yp, lines[i].xp, lines[i].yp) * proDist;

            if (dist(lines[j].xp, lines[j].yp, lines[i].xp, lines[i].yp) < maxDist) {
                ctx.beginPath();
                ctx.lineWidth = 1;
                ctx.shadowBlur = 0;
                ctx.strokeStyle = "hsla(" + (config.hueStart + npt * config.colorSpeed) + ", 50%, 50%, " + (opacity - 50) + "%)"
                ctx.moveTo(lines[i].xp, lines[i].yp);
                ctx.lineTo(lines[j].xp, lines[j].yp);
                ctx.stroke();
                ctx.closePath();
            }
        }

        if (lines[i].xp > w || lines[i].xp < 0 || lines[i].yp > w || lines[i].yp < 0) {
            lines[i] = lines[lines.length - 1];
            lines.pop();
        } else {
            lines[i].xp += lines[i].xv * avg/500;
            lines[i].yp += lines[i].yv * avg/500;
        }
    }
}

function drawSpectrum() {
    noiseSpeed = averageFrequency();
    nrt += noiseSpeed/3000000; // Rotation
    npt += noiseSpeed/1000000; // Distortion
    var avg = 0;

    analyser.getByteFrequencyData(frequencyData);
    var noiseRotate = noise.perlin2(10, nrt);
    var points = Math.round(frequencyDataLen-frequencyDataLen/3);
    var avgFrq = averageFrequency();
    
    for (i = 0; i < points; i++) {
        avg += frequencyData[i];
        avg = avg / points;
        
        var x1 = 
                w / 2 + (config.circleRadius + (avgFrq/4) / points) * 
                Math.cos(-Math.PI/2 + 2 * Math.PI * i / points + noiseRotate);
        var y1 = 
                h / 2 + (config.circleRadius + (avgFrq/4) / points) * 
                Math.sin(-Math.PI/2 + 2 * Math.PI * i / points + noiseRotate);
        var x2 = 
                w / 2 + ((config.circleRadius + (avgFrq/4) / points) + avg * config.multiplier) * 
                Math.cos(-Math.PI/2 + 2 * Math.PI * i / points + noiseRotate);
        var y2 = 
                h / 2 + ((config.circleRadius + (avgFrq/4) / points) + avg * config.multiplier) * 
                Math.sin(-Math.PI/2 + 2 * Math.PI * i / points + noiseRotate);
        var x3 = 
                w / 2 + ((config.circleRadius + (avgFrq/4) / points) + Math.pow((avg * config.multiplier) * 0.09, 2)) * 
                Math.cos(-Math.PI/2 + 2 * Math.PI * i / points + noiseRotate);
        var y3 = 
                h / 2 + ((config.circleRadius + (avgFrq/4) / points) + Math.pow((avg * config.multiplier) * 0.09, 2)) * 
                Math.sin(-Math.PI/2 + 2 * Math.PI * i / points + noiseRotate);
        var nd1 = noise.simplex2(y1/100, npt)*10;
        
        ctx.beginPath();
        ctx.lineCap = "round";
        ctx.shadowBlur = config.glow;
        ctx.lineWidth = 1;
        ctx.strokeStyle = "hsla("+(config.hueStart + npt * config.colorSpeed)+", 50%, "+(20+(Math.pow(avg * 3, 2)))+"%, 100%)";
        ctx.shadowColor = "hsla("+(config.hueStart + npt * config.colorSpeed)+", 50%, "+(20+(Math.pow(avg * 3, 2)))+"%, 100%)";
        ctx.moveTo(x1+nd1, y1+nd1);
        ctx.lineTo(x2+nd1, y2+nd1);
        ctx.stroke();
        ctx.closePath();
        
        ctx.beginPath();
        ctx.lineCap = "round";
        ctx.shadowBlur = config.glow;
        ctx.lineWidth = 4;
        ctx.strokeStyle = "hsla("+(config.hueStart + npt * config.colorSpeed)+", 50%, "+(30+(Math.pow(avg * 3, 2)))+"%, 100%)";
        ctx.shadowColor = "hsla("+(config.hueStart + npt * config.colorSpeed)+", 50%, "+(30+(Math.pow(avg * 3, 2)))+"%, 100%)";
        ctx.moveTo(x1+nd1, y1+nd1);
        ctx.lineTo(x3+nd1, y3+nd1);
        ctx.stroke();
        ctx.closePath();
    }
}

function render() {
    clear();
    drawDots();
    drawSpectrum();
    drawLines();
    requestAnimationFrame(render);
}

function play(uuid) {
    console.log("Playing "+uuid);
    dotEmitter = setInterval(emitDot, 50);
    lineEmitter = setInterval(emitLine, 100);
    bufferSource = null;
    bufferSource = actx.createBufferSource();
    bufferSource.buffer = audioBuffer[uuid];
    bufferSource.loop = false;
    bufferSource.connect(gainNode);
    bufferSource.start();
    render();
}

function lerp(x1, x2, n){
    return x1 + (x2 - x1) * n;
}

function dist(x1, y1, x2, y2) {
    var a = x1 - x2;
    var b = y1 - y2;
    return Math.sqrt(a * a + b * b);
}

function averageFrequency() {
    var avg = 0;
    for (var i = 0; i < frequencyData.length; i++) {
        avg += frequencyData[i];
    }
    return avg;
}

var constraints = {
    audio: true,
    video: false
} 

var AudioContext = window.AudioContext || window.webkitAudioContext;

function uploadRecording(blob) {
    var filename = new Date().toISOString();
    var xhr = new XMLHttpRequest();
    xhr.onload = function(e) {
        if (this.readyState === 4) {
            uuid = e.target.responseText;
            var eventSource = new EventSource("/voicechat/stream?uuid="+uuid);
            eventSource.onmessage = function(e) {
                console.log(e.data);
                if(e.data === "[DONE]") {
                    initializeAudio(uuid);
                    this.close();
                }
            };
        }
    };
    var fd = new FormData();
    fd.append("audio_data", blob, filename);
    xhr.open("POST", "/voicechat/submit", true);
    xhr.send(fd);
}

var gumStream;
var rec;
var audioRecordingContext;
var input;
function toggleRecording() {
    if(recording) {
        recording = false;
        rec.stop();
        document.getElementById("recorder").classList.remove('active');
        gumStream.getAudioTracks()[0].stop();
        rec.exportWAV(uploadRecording);
    } else {
        audioRecordingContext = new AudioContext;
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            gumStream = stream;
            input = audioRecordingContext.createMediaStreamSource(stream);
            rec = new Recorder(input, {numChannels: 1});
            rec.record();
            recording = true;
            document.getElementById("recorder").classList.add('active');
        });
    }
};

document.getElementById("recordAudio").addEventListener("click", toggleRecording);
