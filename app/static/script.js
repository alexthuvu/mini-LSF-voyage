const video = document.getElementById("video");
const button = document.getElementById("toggleCam");
const signExpected= document.getElementById("sign-expected");
const storyPrompt = document.getElementById("story-prompt");
const predictFeedback = document.getElementById("feedback")
const signImage = document.getElementById("sign-image")

const socket = io()

let stream = null;
let isCamOn = false;
let predictionInterval;
let stepIndex = 0;

button.addEventListener("click", async () => {
    console.log("Button clicked!");
    if (!isCamOn) {
        stepIndex = 0;
        signExpected.textContent = "Expected Sign: Bonjour";
        storyPrompt.textContent = 'Bonjour! To start, please click on the button "Turn on camera"\nPour commencer, veuillez cliquer sur le bouton « Activer la caméra »';
        predictFeedback.textContent = "";
        signImage.src = "static/images/bonjour.gif";
        try {
            stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            isCamOn = true;
            button.textContent = "Turn Camera Off / Désactiver la caméra";
            video.onloadedmetadata = () => {
                video.play();
                setTimeout(() => {
                predictionInterval = setInterval(predictWithSocket, 500);
                }, 300);
            };
        }
        catch(err) {
            console.error("Error accessing camera:", err)
                    };
    } else {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }

        isCamOn = false;
        button.textContent = "Turn Camera On / Activer la caméra";
        clearInterval(predictionInterval);
    }
})

function predictWithSocket() {
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
        console.warn("video not ready yet, skipping prediction");
        return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d"); 

    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height); 
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    

    const imageData = canvas.toDataURL("image/jpg");
    socket.emit("predict_frame", {
        image: imageData,
        step_index: stepIndex
    })

}
socket.on("connect", () => {
    console.log("Socket connected");

    socket.on("predict_response", (data) => {
        console.log("Socket result:", data);
        const predicted_label = data.predicted_label;
        stepIndex = data.next_step_index;
        const nextPrompt =  data.next_prompt;
        const nextImage = data.next_image;
        const expected = data.expected;
        const feedback =  data.feedback;
        const leftBox = data.left_box;
        const rightBox = data.right_box;

        predictFeedback.textContent = feedback + " You did " + predicted_label;
        signExpected.textContent = "Expected Sign: " + expected;
        storyPrompt.textContent = "Story Prompt: " + nextPrompt;
        signImage.src = nextImage

        console.log("drawing overlay...")

        const overlay = document.getElementById("overlay");
        const ctxOverlay = overlay.getContext("2d");
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        const canvasWidth = overlay.width;

        ctxOverlay.clearRect(0, 0, overlay.width, overlay.height);
        ctxOverlay.fillStyle = "rgba(0, 0, 0, 0.5)";
        ctxOverlay.fillRect(10, 10, 300, 40);
        ctxOverlay.fillStyle = "white";
        ctxOverlay.font = "16px Tahoma";
        ctxOverlay.fillText(`Prediction: ${predicted_label}`, 20, 35);

        if (leftBox) {
            const [x, y, w, h] = leftBox;
            const flippedX = canvasWidth - x - w;
            ctxOverlay.strokeStyle = "green";
            ctxOverlay.lineWidth = 2;
            ctxOverlay.strokeRect(flippedX, y, w, h);
            }

        if (rightBox) {
            const [x, y, w, h] = rightBox;
            const flippedX = canvasWidth - x - w;
            ctxOverlay.strokeStyle = "green";
            ctxOverlay.lineWidth = 2;
            ctxOverlay.strokeRect(flippedX, y, w, h);
        }

        if (feedback.includes("Correct!")) {
            launchConfetti();
            }


        if (data.story_completed) {
            launchConfetti();
            clearInterval(predictionInterval);

        stepIndex = 0;

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            }

        isCamOn = false;
        button.textContent = "Turn Camera On / Activer la caméra";
        console.log("Story completed! Camera turned off.");
    }
})

})


function launchConfetti() {
    confetti({
        particleCount: 150,
        spread: 70,
        origin: { y: 0.6 }
    });
}

// function captureAndPredict() {
//     if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
//     console.warn("video not ready yet, skipping prediction");
//     return;
//     }
//     const canvas = document.createElement("canvas");
//     canvas.width = video.videoWidth;
//     canvas.height = video.videoHeight;
//     const ctx = canvas.getContext("2d"); 

//     ctx.translate(canvas.width, 0);
//     ctx.scale(-1, 1);

//     ctx.drawImage(video, 0, 0, canvas.width, canvas.height); 
//     ctx.setTransform(1, 0, 0, 1, 0, 0);
    

//     const imageData = canvas.toDataURL("image/jpg");

    
//     fetch("/predict", {
//         method: "POST", 
//         headers: {"Content-Type": "application/json"},
//         body: JSON.stringify({image: imageData, step_index: stepIndex})
//     })
//     .then(res => res.json())
//     .then(data => {
//         const predicted_label = data.predicted_label;
//         stepIndex = data.next_step_index;
//         const nextPrompt =  data.next_prompt;
//         const nextImage = data.next_image;
//         const expected = data.expected;
//         const feedback =  data.feedback;
//         const leftBox = data.left_box;
//         const rightBox = data.right_box;

//         if (nextImage && nextImage !== "undefined") {
//             signImage.src = nextImage;
//         } else {
//             console.warn("Invalid image path received:", nextImage);
//             signImage.src = "app/static/images/question_mark.png";
//         }

//         predictFeedback.textContent = feedback + " You did " + predicted_label;
//         signExpected.textContent = "Expected Sign: " + expected;
//         storyPrompt.textContent = "Story Prompt: " + nextPrompt;
//         signImage.src = nextImage

//         const overlay = document.getElementById("overlay");
//         const ctxOverlay = overlay.getContext("2d");
//         overlay.width = video.videoWidth;
//         overlay.height = video.videoHeight;
//         const canvasWidth = overlay.width;

//         ctxOverlay.clearRect(0, 0, overlay.width, overlay.height);
//         ctxOverlay.fillStyle = "rgba(0, 0, 0, 0.5)";
//         ctxOverlay.fillRect(10, 10, 300, 40);
//         ctxOverlay.fillStyle = "white";
//         ctxOverlay.font = "16px Tahoma";
//         ctxOverlay.fillText(`Prediction: ${predicted_label}`, 20, 35);

//         if (leftBox) {
//             const [x, y, w, h] = leftBox;
//             const flippedX = canvasWidth - x - w;
//             ctxOverlay.strokeStyle = "green";
//             ctxOverlay.lineWidth = 2;
//             ctxOverlay.strokeRect(flippedX, y, w, h);
//         }

//         if (rightBox) {
//             const [x, y, w, h] = rightBox;
//             const flippedX = canvasWidth - x - w;
//             ctxOverlay.strokeStyle = "green";
//             ctxOverlay.lineWidth = 2;
//             ctxOverlay.strokeRect(flippedX, y, w, h);
//         }

//         if (feedback.includes("Correct!")) {
//             launchConfetti();
//         }


//         if (data.story_completed) {
//             launchConfetti();
//             clearInterval(predictionInterval);

//             stepIndex = 0;

//             if (stream) {
//                 stream.getTracks().forEach(track => track.stop());
//                 video.srcObject = null;
//             }

//             isCamOn = false;
//             button.textContent = "Turn Camera On / Activer la caméra";
//             console.log("Story completed! Camera turned off.");
//         }


//     })
//     .catch(err => {
//     console.error("Prediction error:", err);
// });

// }


