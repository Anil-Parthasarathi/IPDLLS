const vertexShaderCode = `
    attribute vec4 pos;
    
    void main() {
    
        gl_Position = pos;
    }
`;

const finalFragmentShaderCode = `

precision mediump float;

uniform int renderStyle;
uniform float zFactor;

uniform float DEPTHFACTOR;

uniform vec3 iResolution;
uniform vec4 iMouse;

uniform sampler2D depth;
uniform sampler2D image;

/*mat3 sobelFilterX = mat3(
    -1.0,  0.0,  1.0, 
    -2.0,  0.0,  2.0, 
    -1.0,  0.0,  1.0
);

mat3 sobelFilterY = mat3(
    -1.0,  -2.0,  -1.0, 
    0.0,  0.0,  0.0, 
    1.0,  2.0,  1.0
);*/

void main() {

    vec2 uv = gl_FragCoord.xy / iResolution.xy; // Normalized pixel coordinates

    vec4 disparityMap = texture2D(depth, uv);
    vec4 originalImage = texture2D(image, uv);

    if (renderStyle == 4){
        gl_FragColor = disparityMap;
        return;
    }

    vec3 specular = vec3(1.0, 1.0, 0.9);
    vec3 ambient = vec3(0.10, 0.10, 0.60);
    vec3 diffuse = vec3(1.00, 0.80, 0.50);
    
    ambient = (0.1 * ambient + originalImage.xyz) / 1.1;
    diffuse = 0.8 * (0.1 * diffuse + originalImage.xyz) / 1.1; // added 0.8 in front to dim the color of the image so that effects of light are more pronounced

    vec3 lightpos = vec3(iMouse.x, iMouse.y, 70.0); //adjust this to change the size, closeness, of light
    vec3 dir = normalize(lightpos - vec3(gl_FragCoord.xy, 0.0));

    ///////////////////////////////////////////////////////////

    //computing the normal map

    //first compute the gradient by convolving through with a sobel filter

    //need to have boundary conditions to ensure not going out of bounds

    float gradX = 0.0;
    float gradY = 0.0;

    vec2 texel = 1.0 / iResolution.xy; 

    if (uv.x > 0.0){

        vec2 neighborFragment = vec2(uv.x - texel.x, uv.y);

        gradX += -2.0 * (DEPTHFACTOR / max(texture2D(depth,  neighborFragment).r, 0.0001));
        
    }

    if (uv.x + texel.x < 1.0){

        vec2 neighborFragment = vec2(uv.x + texel.x, uv.y);

        gradX += 2.0 * (DEPTHFACTOR / max(texture2D(depth,  neighborFragment).r, 0.0001));

    }

    if (uv.y > 0.0){

        vec2 neighborFragment = vec2(uv.x, uv.y - texel.y);

        gradY += -2.0 * (DEPTHFACTOR / max(texture2D(depth,  neighborFragment).r, 0.0001));

        if (uv.x > 0.0){

            vec2 neighborFragment2 = vec2(uv.x - texel.x, neighborFragment.y);

            float gradVal = -1.0 * texture2D(depth,  neighborFragment2).r;

            gradX += gradVal;
            gradY += gradVal;
            
        }

        if (uv.x + texel.x < 1.0){

            vec2 neighborFragment2 = vec2(uv.x + texel.x, neighborFragment.y);

            gradX += 1.0 * texture2D(depth,  neighborFragment2).r;
            gradY += -1.0 * texture2D(depth,  neighborFragment2).r;
            
        }
    }

    if (uv.y + texel.y < 1.0){

        vec2 neighborFragment = vec2(uv.x, uv.y + texel.y);

        gradY += 2.0 * (DEPTHFACTOR / max(texture2D(depth,  neighborFragment).r, 0.0001));

        if (uv.x > 0.0){

            vec2 neighborFragment2 = vec2(uv.x - texel.x, neighborFragment.y);

            gradX += -1.0 * texture2D(depth,  neighborFragment2).r;
            gradY += 1.0 * texture2D(depth,  neighborFragment2).r;
            
        }

        if (uv.x + texel.x < 1.0){

            vec2 neighborFragment2 = vec2(uv.x + texel.x, neighborFragment.y);

            float gradVal = 1.0 * texture2D(depth,  neighborFragment2).r;

            gradX += gradVal;
            gradY += gradVal;
            
        }
    }

    vec3 norm = normalize(vec3(-1.0 * gradX, -1.0 * gradY, zFactor)); //adjust z value here to change depth effect, this is what I thought looked best but might want to experiment more

    /////////////////////////////////////////////////////////////

    //if render style is 1 then we will just return the normal map

    if (renderStyle == 1){
        gl_FragColor = vec4(norm, 1.0);
        return;
    }

    //vec3 norm = normalize(2.0 * normal - vec3(1.0));

    vec3 eye = vec3(0.0, 0.0, 1.0);

    if (renderStyle == 2){
        //silhouette shader

        float normEye = dot(norm, eye);

        if (abs(normEye) < 0.3){

		gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);

        }
        else{
            gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }

        return;

    }
        
    vec3 cd = diffuse * max(0.0, dot(dir, norm)); //diffuse color
	
	vec3 h = normalize(dir + eye);

	vec3 cs = specular * pow(max(0.0, dot(h, norm)), 100.0); //specular color

    vec3 col = ambient + cd + cs;

    if (renderStyle == 3){
        // cel shader

        float normEye = dot(norm, eye);

        if (abs(normEye) < 0.2){
		    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
        else if (abs(normEye) < 0.3) {
            gl_FragColor = vec4(col.x * 0.3, col.y *  0.2, col.z *  0.25, 1.0);
        }
        else if (abs(normEye) < 0.4) {
            gl_FragColor = vec4(col.x * 0.5, col.y *  0.4, col.z *  0.45, 1.0);
        }
        else if (abs(normEye) < 0.7) {
            gl_FragColor = vec4(col.x * 0.65, col.y *  0.55, col.z *  0.6, 1.0);
        }
        else if (abs(normEye) < 0.8) {
            gl_FragColor = vec4(col.x * 0.8, col.y *  0.7, col.z *  0.75, 1.0);
        }
        else if (abs(normEye) < 0.9) {
            gl_FragColor = vec4(col.x * 0.9, col.y *  0.85, col.z *  0.9, 1.0);
        }
        else if (abs(normEye) < 1.0) {
            gl_FragColor = vec4(col.x * 0.95, col.y *  0.9, col.z *  0.95, 1.0);
        }
        else{
            gl_FragColor = vec4(col.x, col.y, col.z, 1.0);
        }
        return;

    }
    
    gl_FragColor = vec4(col, 1.0); // Output to screen
}
`;

var canv;
var glContext;
var glProgram;

var mousePos;
var iResolutionAttribute;
var iMouseAttribute;

var renderStyle = 0;
var renderStyleAttribute;

var depthFactor = 598400.0;
var zFactor = 0.15;
var zFactorAttribute;
var depthFactorAttribute;

async function main() {

    //set up the canvas
    canv = document.querySelector("#renderCanvas");

    glContext = canv.getContext("webgl");

    //check if context was sucessfully retrieved
    if (glContext == null){
        print("CONTEXT ERROR");
        return;
    }

    //setup webgl program
    glProgram = glContext.createProgram();

    //set shaders
    var vertexShader = glContext.createShader(glContext.VERTEX_SHADER);
    var fragmentShader = glContext.createShader(glContext.FRAGMENT_SHADER);
    
    glContext.shaderSource(vertexShader, vertexShaderCode);
    glContext.shaderSource(fragmentShader, finalFragmentShaderCode);

    //compile the shaders
    glContext.compileShader(vertexShader);
    glContext.compileShader(fragmentShader);

    var success = glContext.getShaderParameter(fragmentShader, glContext.COMPILE_STATUS);

    if (!success){
        throw ("Issue with compiling:" + glContext.getShaderInfoLog(fragmentShader));
    }

    //attach shaders
    glContext.attachShader(glProgram, vertexShader);
    glContext.attachShader(glProgram, fragmentShader);

    //link the program
    glContext.linkProgram(glProgram);

    glContext.useProgram(glProgram);

    //need to do this to prevent textures from being upside down!
    glContext.pixelStorei(glContext.UNPACK_FLIP_Y_WEBGL, true);

    window.getDisparity();

    //set up a rectangle to use as a frame to render with
    const verts = new Float32Array([
        -1, -1,   
        1, -1,   
        -1,  1,

        -1,  1,   
        1, -1,    
        1,  1
    ]);

    //create and bind the vertex buffer
    const vertexBuffer = glContext.createBuffer();
    glContext.bindBuffer(glContext.ARRAY_BUFFER, vertexBuffer);
    glContext.bufferData(glContext.ARRAY_BUFFER, verts, glContext.STATIC_DRAW);

    //assign position in the shader
    const posAttributeLoc = glContext.getAttribLocation(glProgram, "pos");
    glContext.enableVertexAttribArray(posAttributeLoc);
    glContext.vertexAttribPointer(posAttributeLoc, 2, glContext.FLOAT, false, 0, 0);

    //add mouse and resolution variables to the shader
    iMouseAttribute = glContext.getUniformLocation(glProgram, 'iMouse');
    iResolutionAttribute = glContext.getUniformLocation(glProgram, 'iResolution');

    renderStyleAttribute = glContext.getUniformLocation(glProgram, 'renderStyle');
    depthFactorAttribute = glContext.getUniformLocation(glProgram, 'DEPTHFACTOR');
    zFactorAttribute = glContext.getUniformLocation(glProgram, 'zFactor');

    //set up the mouse and an event listener for it
    mousePos = { x: 0, y: 0 };

    canv.addEventListener('mousemove', (event) => {

        //upon a mouse movement event move the mouse position

        const rect = canv.getBoundingClientRect();

        const xCanvScale = canv.width / rect.width;
        const yCanvScale = canv.height / rect.height;

        mousePos.x = (event.clientX - rect.left) * xCanvScale;
        mousePos.y = canv.height - (event.clientY - rect.top) * yCanvScale;

    });
    
    render();
}

function render(){
    glContext.clearColor(0.0, 0.0, 0.0, 1.0); //setting background color to black initially

    glContext.clear(glContext.COLOR_BUFFER_BIT);

    //before starting to draw the new frame, update mouse and resolution values in the shader

    glContext.uniform4f(iMouseAttribute, mousePos.x, mousePos.y, 0.0, 0.0);
    glContext.uniform3f(iResolutionAttribute, canv.width, canv.height, 1.0);

    glContext.uniform1i(renderStyleAttribute, renderStyle);
    glContext.uniform1f(zFactorAttribute, zFactor);
    glContext.uniform1f(depthFactorAttribute, depthFactor);

    //draw!
    glContext.drawArrays(glContext.TRIANGLES, 0, 6);
    requestAnimationFrame(render);
}

//create a texture and set up the texture attributes which can then be sent to the shader
async function textureLoader(url) {

    return new Promise((resolve) => {

        const newTexture = glContext.createTexture();
        const textureImage = new Image();

        textureImage.onload = () => {

            glContext.bindTexture(glContext.TEXTURE_2D, newTexture);

            glContext.texImage2D(glContext.TEXTURE_2D, 0, glContext.RGBA, glContext.RGBA, glContext.UNSIGNED_BYTE, textureImage);

            glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_WRAP_T, glContext.CLAMP_TO_EDGE);
            glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_WRAP_S, glContext.CLAMP_TO_EDGE);
            glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_MIN_FILTER, glContext.LINEAR);
            
            resolve(newTexture);
        };

        textureImage.src = url;
    });
}

async function setRenderStyle(style){
    if (style == "0"){
        renderStyle = 0;
    }
    else if (style == "1"){
        renderStyle = 1;
    }
    else if (style == "2"){
        renderStyle = 2;
    }
    else if (style == "3"){
        renderStyle = 3;
    }
    else if (style == "4"){
        renderStyle = 4;
    }
}
window.setRenderStyle = setRenderStyle;

export function handleZSliderChange(newZFactor) {
    document.getElementById("sliderValue").textContent = newZFactor;
    
    zFactor = newZFactor;

}
window.handleZSliderChange = handleZSliderChange;

export function handleDepthFactorChange(newDepthFactor) {
    document.getElementById("depthSlider").textContent = newDepthFactor;
    
    depthFactor = newDepthFactor;

}
window.handleDepthFactorChange = handleDepthFactorChange;

window.updateTexture = async function updateTexture(image, depth) {
    const textures = await Promise.all([image,depth].map(textureLoader));

    for (var i = 0; i < textures.length; i++) {
        glContext.activeTexture(glContext.TEXTURE0 + i);
        glContext.bindTexture(glContext.TEXTURE_2D, textures[i]);
    }

    //setup texture attributes
    //right now just passing in normal and image but later we can get this setup for multiple passes

    const iChannel0Loc = glContext.getUniformLocation(glProgram, "depth");
    glContext.uniform1i(iChannel0Loc, 0);

    const iChannel1Loc = glContext.getUniformLocation(glProgram, "image");
    glContext.uniform1i(iChannel1Loc, 1);
}


main();