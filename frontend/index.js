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
    const left = document.createElement('img');
    left.src = `../images/${imageDir}/left.png`;
    const right = document.createElement('img');
    right.src = `../images/${imageDir}/right.png`;

    const imageInput = document.createElement('input');
    imageInput.type = 'radio';
    imageInput.id = imageDir;
    imageInput.name = 'images';

    const imageLabel = document.createElement('label');
    imageLabel.htmlFor = imageDir;
    imageLabel.append(left, right);

    const imageDiv = document.createElement('div');
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
form.addEventListener('submit', () => selectImage());
