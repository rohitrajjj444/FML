const get = id => document.getElementById(id);
const video = get('camera');
const preview = get('photo-preview');
const request = get('request');
const photos = [null, null];
const steps = [...document.querySelectorAll('.steps li')];
const prompts = [
  'A breathable linen outfit for summer',
  'A black shirt for a birthday party',
  'A cotton kurta for a wedding',
  'A smart casual outfit for the office',
  'A denim jacket with relaxed trousers',
  'A colourful outfit for a festival',
];
const photoSteps = [
  ['Take a face photo', 'Look straight at the camera in even lighting.', 'Your face photo'],
  ['Take a full-body photo', 'Step back so your head and feet fit in the frame.', 'Your full-body photo'],
];

let step = 0;
let stream;
let cameraAttempt = 0;
let selectionAttempt = 0;

function updatePreview() {
  const live = Boolean(stream);
  const photo = photos[step];
  video.hidden = !live;
  preview.hidden = live || !photo;
  get('placeholder').hidden = live || Boolean(photo);
  get('open-camera').hidden = live;
  get('capture').hidden = !live;
  get('close-camera').hidden = !live;
  get('photo-next').disabled = !photo || live;
  if (photo) preview.src = photo;
  else preview.removeAttribute('src');
}

function stopCamera() {
  cameraAttempt += 1;
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  get('open-camera').disabled = false;
  get('capture').disabled = true;
  updatePreview();
}

function showStep(next, focus = true) {
  selectionAttempt += 1;
  stopCamera();
  step = next;
  steps.forEach((item, index) => {
    if (index === Math.min(step, 2)) item.setAttribute('aria-current', 'step');
    else item.removeAttribute('aria-current');
  });
  get('photo-step').hidden = step > 1;
  get('request-step').hidden = step !== 2;
  get('review').hidden = step !== 3;
  if (step < 2) {
    const [title, tip, alt] = photoSteps[step];
    get('photo-title').textContent = title;
    get('photo-tip').textContent = tip;
    preview.alt = alt;
    get('photo-back').hidden = step === 0;
    get('photo-status').textContent = '';
    updatePreview();
  }
  if (focus) get(step < 2 ? 'photo-title' : step === 2 ? 'request-title' : 'review-title').focus();
}

function setPhoto(blob) {
  if (photos[step]) URL.revokeObjectURL(photos[step]);
  photos[step] = URL.createObjectURL(blob);
  stopCamera();
  get('photo-status').textContent = 'Photo ready. You can replace it or continue.';
}

get('open-camera').addEventListener('click', async () => {
  const attempt = ++cameraAttempt;
  selectionAttempt += 1;
  get('open-camera').disabled = true;
  get('photo-status').textContent = 'Waiting for camera access…';
  try {
    const camera = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    if (attempt !== cameraAttempt) {
      camera.getTracks().forEach(track => track.stop());
      return;
    }
    stream = camera;
    video.srcObject = stream;
    updatePreview();
    get('photo-status').textContent = '';
  } catch {
    if (attempt !== cameraAttempt) return;
    stopCamera();
    get('photo-status').textContent = 'Camera unavailable. Check browser permission or upload a photo.';
  }
});

video.addEventListener('loadeddata', () => { get('capture').disabled = !stream || !video.videoWidth; });
get('close-camera').addEventListener('click', () => {
  selectionAttempt += 1;
  stopCamera();
  get('photo-status').textContent = '';
});
get('capture').addEventListener('click', () => {
  if (!stream || !video.videoWidth) return;
  const attempt = ++selectionAttempt;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(blob => {
    if (attempt !== selectionAttempt) return;
    if (blob) setPhoto(blob);
    else get('photo-status').textContent = 'Could not capture the photo. Try again or upload a photo.';
  }, 'image/jpeg', 0.9);
});

get('upload').addEventListener('change', async event => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  const attempt = ++selectionAttempt;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 12 * 1024 * 1024) {
    get('photo-status').textContent = 'Choose a JPEG, PNG or WebP photo up to 12 MB.';
    return;
  }
  try {
    const image = await createImageBitmap(file);
    image.close();
    if (attempt === selectionAttempt) setPhoto(file);
  } catch {
    if (attempt === selectionAttempt) get('photo-status').textContent = 'This image could not be opened. Try another photo.';
  }
});

function updateSuggestions() {
  const words = request.value.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  const matches = words.length ? prompts
    .map(prompt => ({
      prompt,
      score: words.filter(word => prompt.toLowerCase().split(' ').some(part => part.startsWith(word))).length,
    }))
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .map(item => item.prompt) : prompts;
  const choices = matches.filter(prompt => prompt.toLowerCase() !== request.value.trim().toLowerCase()).slice(0, 3);
  get('suggestion-label').hidden = choices.length === 0;
  get('suggestions').replaceChildren(...choices.map(prompt => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = prompt;
    button.addEventListener('click', () => {
      request.value = prompt;
      request.setCustomValidity('');
      request.focus();
      updateSuggestions();
    });
    return button;
  }));
}

request.addEventListener('input', () => {
  request.setCustomValidity('');
  updateSuggestions();
});
get('photo-next').addEventListener('click', () => {
  if (photos[step] && !stream) showStep(step + 1);
});
get('photo-back').addEventListener('click', () => showStep(0));
get('request-back').addEventListener('click', () => showStep(1));
get('review-back').addEventListener('click', () => showStep(2));
get('request-step').addEventListener('submit', event => {
  event.preventDefault();
  const text = request.value.trim();
  if (!text) {
    request.setCustomValidity('Describe what you want to wear.');
    request.reportValidity();
    return;
  }
  get('review-face').src = photos[0];
  get('review-body').src = photos[1];
  get('review-request').textContent = text;
  showStep(3);
});
get('reset').addEventListener('click', () => {
  photos.forEach(photo => { if (photo) URL.revokeObjectURL(photo); });
  photos.fill(null);
  get('review-face').removeAttribute('src');
  get('review-body').removeAttribute('src');
  get('review-request').textContent = '';
  request.value = '';
  request.setCustomValidity('');
  updateSuggestions();
  showStep(0);
});
window.addEventListener('pagehide', stopCamera);

updateSuggestions();
showStep(0, false);

const sendButton = get('send-request');
const sendStatus = get('send-status');
const outfitResults = get('outfit-results');
const outfitList = get('outfit-list');

get('request-step').addEventListener('submit', () => {
  sendStatus.textContent = '';
  get('style-profile').hidden = true;
  outfitResults.hidden = true;
});

sendButton.addEventListener('click', async () => {
  const reviewBack = get('review-back');
  const resetButton = get('reset');

  sendButton.disabled = true;
  reviewBack.disabled = true;
  resetButton.disabled = true;
  outfitResults.hidden = true;
  outfitList.replaceChildren();
  get('style-profile').hidden = true;
  sendStatus.textContent = 'Sending your request…';

  try {
    const [face, body] = await Promise.all(
      photos.map(async url => (await fetch(url)).blob())
    );

    const form = new FormData();
    form.append('face_photo', face, 'face-photo');
    form.append('body_photo', body, 'body-photo');
    form.append('clothing_request', get('review-request').textContent);

    const response = await fetch('http://localhost:8001/analyze', {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(30_000),
    });

    const result = await response.json();
    if (!response.ok) {
      const message = typeof result.detail === 'string'
        ? result.detail
        : 'Check that you selected both photos and entered a clothing request.';
      throw new Error(message);
    }

    const { label, confidence } = result.face_shape;
    const percentage = (confidence * 100).toFixed(2);
    sendStatus.textContent = `Predicted face shape: ${label} (${percentage}% model confidence).`;

    const bodyRatio = result.shoulder_to_hip_landmark_ratio;
    if (typeof bodyRatio === 'number') {
      sendStatus.append(` Shoulder-to-hip landmark ratio: ${bodyRatio.toFixed(2)}.`);
    } else {
      sendStatus.append(' Body ratio unavailable. Try a straight-on photo with both shoulders and hips in frame.');
    }

    const swatch = result.skin_colour_sample;
    if (typeof swatch === 'string' && /^#[0-9a-f]{6}$/i.test(swatch)) {
      const chip = document.createElement('span');
      chip.style.backgroundColor = swatch;
      chip.style.color = '#fff';
      chip.style.textShadow = '0 1px 3px #000';
      chip.style.padding = '2px 6px';
      chip.textContent = swatch;
      sendStatus.append(' Photo colour sample: ', chip);
    }

    const ideas = result.recommendations ?? [];
    
    get('profile-face').value = label.toLowerCase();
    get('profile-body').value = '';
    get('profile-skin').value = '';
    get('style-profile').hidden = false;

    if (ideas.length === 0) {
      const item = document.createElement('li');
      item.textContent = 'No matches in the sample catalogue. Try a broader request, such as “kurta” instead of a specific colour.';
      outfitList.append(item);
    }

    for (const idea of ideas) {
      const item = document.createElement('li');
      const name = document.createElement('strong');
      name.textContent = idea.name;
      const tip = document.createElement('p');
      tip.textContent = idea.styling_tip;
      const match = document.createElement('small');
      match.textContent = `Matches: ${idea.matched_keywords.join(', ')}`;
      item.append(name, tip, match);

      const ratingControls = document.createElement('div');
      const ratingStatus = document.createElement('p');
      ratingStatus.setAttribute('role', 'status');

      for (const rating of ['like', 'dislike']) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = rating === 'like' ? 'Like' : 'Dislike';
        ratingControls.append(button);

        button.addEventListener('click', async () => {
          const buttons = ratingControls.querySelectorAll('button');
          buttons.forEach(control => { control.disabled = true; });
          ratingStatus.textContent = 'Saving…';

          try {
            const response = await fetch('http://localhost:8001/feedback', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                clothing_request: get('review-request').textContent,
                outfit: idea.name,
                face_shape: get('profile-face').value,
                body_shape: get('profile-body').value,
                skin_depth: get('profile-skin').value,
                rating,
              }),
              signal: AbortSignal.timeout(10_000),
            });

            if (!response.ok) throw new Error('Feedback could not be saved.');
            ratingStatus.textContent = `Saved: ${rating}.`;
          } catch {
            ratingStatus.textContent = 'Could not confirm the save. Check the backend.';
            buttons.forEach(control => { control.disabled = false; });
          }
        });
      }

      item.append(name, tip, match, ratingControls, ratingStatus);
      if (typeof idea.image_url === 'string' && idea.image_url.startsWith('https://')) {
        const image = document.createElement('img');
        image.className = 'outfit-product-image';
        image.src = idea.image_url;
        image.alt = idea.product_name || 'Featured garment';
        image.loading = 'lazy';
        image.addEventListener('error', () => image.remove(), { once: true });
        item.prepend(image);
      }
      if (Number.isFinite(idea.listed_price_inr) && idea.listed_price_inr > 0) {
        const price = document.createElement('p');
        const amount = new Intl.NumberFormat('en-IN', {
          style: 'currency', currency: 'INR', maximumFractionDigits: 0,
        }).format(idea.listed_price_inr);
        price.textContent = `Listed price: ${amount}. Check retailer for the current price.`;
        item.insertBefore(price, ratingControls);
      }
      if (typeof idea.product_url === 'string' && idea.product_url.startsWith('https://')) {
        const link = document.createElement('a');
        link.className = 'outfit-product-link';
        link.href = idea.product_url;
        const productName = idea.product_name || 'product';
        const retailer = idea.retailer ? ` at ${idea.retailer}` : '';
        link.textContent = `View ${productName}${retailer} ↗`;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        item.insertBefore(link, ratingControls);
      }

      outfitList.append(item);
    }

    outfitResults.hidden = false;
  } catch (error) {
    if (error.name === 'TimeoutError') {
      sendStatus.textContent = 'Analysis timed out after 30 seconds. Please try again.';
    } else if (error instanceof TypeError) {
      sendStatus.textContent = 'Could not reach the backend. Check that it is running on port 8001.';
    } else {
      sendStatus.textContent = error.message;
    }
  } finally {
    sendButton.disabled = false;
    reviewBack.disabled = false;
    resetButton.disabled = false;
  }
});
