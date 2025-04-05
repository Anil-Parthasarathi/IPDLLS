function addImages() {
  const imageFiles = [
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

  imageFiles.forEach((dirName) => {
    const left = document.createElement('img');
    left.src = `../images/${dirName}/left.png`;
    const right = document.createElement('img');
    right.src = `../images/${dirName}/right.png`;

    const imageDiv = document.createElement('div');

    const imageInput = document.createElement('input');
    imageInput.type = 'radio';
    imageInput.id = dirName;
    imageInput.name = 'images';

    const imageLabel = document.createElement('label');
    imageLabel.append(left, right);
    imageLabel.htmlFor = dirName;

    imageDiv.append(imageInput, imageLabel);

    imageContainer.append(imageDiv);
  });
}

document.body.onload = addImages();

async function selectImage() {
  const selected = document.querySelector('input[name="images"]:checked');
  if (!selected) {
    return;
  }

  const imageDir = selected.id;
  console.log(imageDir);

  try {
    const response = await fetch(
      `http://127.0.0.1:8000/disparity?name=${imageDir}`
    );
    if (!response.ok) {
      alert('Nonvalid images.');
    }

    const image = await response.blob();
    const imageUrl = URL.createObjectURL(image);

    const testImage = document.getElementById('test');
    testImage.src = imageUrl;
  } catch (error) {
    console.error(error);
  }
}

const form = document.getElementsByTagName('form')[0];
form.addEventListener('submit', (event) => selectImage(event));
