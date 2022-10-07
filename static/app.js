var imgData;
function keyPressed() {
    const img = get().drawingContext.getImageData(0, 0, 280, 280)
    imgData = {
        data: img.data,
        width: img.width,
        height: img.height
    };
    imgDataJson = JSON.stringify(imgData)
    
    post_img(imgDataJson);
}

function post_img(data) {
    
    var xhr = new XMLHttpRequest();
    var url = "/postimage";
    xhr.open('POST', url, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(data)
}