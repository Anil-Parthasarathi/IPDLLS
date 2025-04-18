function addImages() {
  const imageDirs = [
    'art',
    'books',
    'computer',
    'dolls',
    'drumsticks',
    'dwarves',
    'laundry',
    'moebius',
    'reindeer',
  ];

  const imageContainer = document.getElementsByTagName('fieldset')[0];

  imageDirs.forEach((imageDir) => {
    const imageInput = document.createElement('input');
    imageInput.type = 'radio';
    imageInput.id = imageDir;
    imageInput.name = 'images';

    const imageLabel = document.createElement('label');
    imageLabel.htmlFor = imageDir;
    imageLabel.innerText = imageDir[0].toUpperCase() + imageDir.slice(1);

    const imageDiv = document.createElement('div');
    imageDiv.append(imageInput, imageLabel);

    imageContainer.append(imageDiv);
  });

  imageContainer.firstChild.firstChild.checked = true;
  imageContainer.addEventListener('change', (event) => {
    if (event.target && event.target.type === 'radio') {
      console.log(event.target.id);
      selectImage(event.target.id);
    }
  });

  selectImage(imageDirs[0]);
}

document.body.onload = addImages();

function selectImage(imageDir) {
  const leftImage = document.getElementById('selected-left');
  const rightImage = document.getElementById('selected-right');

  leftImage.src = `../images/${imageDir}/left.png`;
  rightImage.src = `../images/${imageDir}/right.png`;
}

async function getDisparity() {
  const selected = document.querySelector('input[name="images"]:checked');
  if (!selected) {
    return;
  }

  const imageDir = selected.id;

  try {
    const response = await fetch(
      `http://127.0.0.1:8000/disparity?name=${imageDir}`
    );

    if (!response.ok) {
      alert('Nonvalid images.');
    }

    // Send disparity image to render pipeline
    const image = await response.blob();
  } catch (error) {
    console.error(error);
  }
}

const form = document.getElementsByTagName('form')[0];
form.addEventListener('submit', () => getDisparity());
