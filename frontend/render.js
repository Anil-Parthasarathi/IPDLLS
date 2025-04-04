const vertexShaderCode = `
    attribute vec4 pos;
    
    void main() {
    
        gl_Position = pos;
    }
`;

const finalFragmentShaderCode = `
precision mediump float;

uniform vec3 iResolution;
uniform vec4 iMouse;

uniform sampler2D normal;
uniform sampler2D image;

void main() {

    vec2 uv = gl_FragCoord.xy / iResolution.xy; // Normalized pixel coordinates

    vec4 normalMap = texture2D(normal, uv);
    vec4 originalImage = texture2D(image, uv);

    vec3 specular = vec3(1.0, 1.0, 0.9);
    vec3 ambient = vec3(0.10, 0.10, 0.60);
    vec3 diffuse = vec3(1.00, 0.80, 0.50);
    
    ambient = (0.1 * ambient + originalImage.xyz) / 1.1;
    diffuse = (0.1 * diffuse + originalImage.xyz) / 1.1;

    vec3 lightpos = vec3(iMouse.x, iMouse.y, 25.0);
    vec3 dir = normalize(lightpos - vec3(gl_FragCoord.xy, 0.0));

    vec3 norm;
    norm = normalize(2.0 * normalMap.rgb - vec3(1.0));

    vec3 eye = vec3(0.0, 0.0, 1.0);
        
    vec3 cd = diffuse * max(0.0, dot(dir, norm)); //diffuse color
	
	vec3 h = normalize(dir + eye);

	vec3 cs = specular * pow(max(0.0, dot(h, norm)), 50.0); //specular color

    vec3 col = ambient + cd + cs;
    
    gl_FragColor = vec4(col, 1.0); // Output to screen
}
`;

var canv;
var glContext;
var glProgram;

var mousePos;
var iResolutionAttribute;
var iMouseAttribute;

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

    const textureAssets = [
        'Assets/normal.png',
        'Assets/image.png',
    ];

    const textures = await Promise.all(textureAssets.map(textureLoader));

    for (var i = 0; i < textures.length; i++) {
        glContext.activeTexture(glContext.TEXTURE0 + i);
        glContext.bindTexture(glContext.TEXTURE_2D, textures[i]);
    }

    //setup texture attributes
    //right now just passing in normal and image but later we can get this setup for multiple passes

    const iChannel0Loc = glContext.getUniformLocation(glProgram, "normal");
    glContext.uniform1i(iChannel0Loc, 0);

    const iChannel1Loc = glContext.getUniformLocation(glProgram, "image");
    glContext.uniform1i(iChannel1Loc, 1);

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


main();