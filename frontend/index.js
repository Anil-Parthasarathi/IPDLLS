// Adds inputs and image containers for selection of image pairs
function addImages() {
  const imageNames = [
    'reindeer',
    'art',
    'books',
    'computer',
    'dolls',
    'drumsticks',
    'dwarves',
    'laundry',
    'moebius',
  ];

  const imageContainer = document.getElementsByTagName('fieldset')[0];

  imageNames.forEach((imageName) => {
    const imageInput = document.createElement('input');
    imageInput.type = 'radio';
    imageInput.id = imageName;
    imageInput.name = 'images';

    const imageLabel = document.createElement('label');
    imageLabel.htmlFor = imageName;
    imageLabel.innerText = imageName[0].toUpperCase() + imageName.slice(1);

    const imageDiv = document.createElement('div');
    imageDiv.append(imageInput, imageLabel);

    imageContainer.append(imageDiv);
  });

  imageContainer.firstChild.firstChild.checked = true;

  // Create callback to change displayed image pair
  imageContainer.addEventListener('change', (event) => {
    if (event.target && event.target.type === 'radio') {
      selectImage(event.target.id);
    }
  });

  selectImage(imageNames[0]);
}

document.body.onload = addImages();

// Displays selected image pair
function selectImage(imageName) {
  const leftImage = document.getElementById('selected-left');
  const rightImage = document.getElementById('selected-right');

  leftImage.src = `../images/${imageName}/left.png`;
  rightImage.src = `../images/${imageName}/right.png`;
}

// Updates canvas with selected image pair and disparity map
window.getDisparity = async function getDisparity() {
  const selected = document.querySelector('input[name="images"]:checked');
  if (!selected) {
    return;
  }
  
  const imageName = selected.id;

  textureAssets = [
    `../images/${imageName}/depth.png`,
    `../images/${imageName}/left.png`,
  ]

  window.updateTexture(`../images/${imageName}/depth.png`, `../images/${imageName}/left.png`)
}

const form = document.getElementsByTagName('form')[0];
form.addEventListener('submit', () => getDisparity());
