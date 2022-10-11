
var prev_pos;

function setup() {

    const cnv = createCanvas(280, 280);


    background(0);

}
function draw() {

    // Drawing
    if (mouseIsPressed) {
        if (!prev_pos) { prev_pos = [mouseX, mouseY]; }

        stroke(255);
        strokeWeight(20);
        line(prev_pos[0], prev_pos[1], mouseX, mouseY);
        prev_pos = [mouseX, mouseY];
    }
    else {
        prev_pos = undefined;
    }
    
}



