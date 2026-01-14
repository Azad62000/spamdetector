function showTab(tabName) {
  const tabs = document.querySelectorAll('.tab-content');
  const buttons = document.querySelectorAll('.tab-button');
  tabs.forEach(tab => tab.classList.remove('active'));
  buttons.forEach(button => button.classList.remove('active'));
  document.getElementById(tabName + '-tab').classList.add('active');
  event.target.classList.add('active');
}

document.getElementById('text-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const text = document.getElementById('email-text').value;
  await predict(text);
});

document.getElementById('file-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const file = document.getElementById('email-file').files[0];
  if (file) {
    const formData = new FormData();
    formData.append('file', file);
    await predictFile(formData);
  }
});

async function predict(text) {
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text }),
    });
    const data = await response.json();
    displayResult(data);
  } catch (error) {
    alert('An error occurred while predicting. Please try again.');
  }
}

async function predictFile(formData) {
  try {
    const response = await fetch('/predict_file', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    displayResult(data);
  } catch (error) {
    alert('An error occurred while predicting. Please try again.');
  }
}

function displayResult(data) {
  const resultDiv = document.getElementById('result');
  const resultText = document.getElementById('result-text');
  const probabilityBar = document.getElementById('probability-bar');
  const probabilityFill = document.getElementById('probability-fill');
  const probabilityText = document.getElementById('probability-text');
  resultDiv.classList.remove('hidden');
  if (data.label === 'spam') {
    resultText.innerHTML = '<span style="color: #ef4444; font-weight: bold;">This email is classified as SPAM</span>';
  } else {
    resultText.innerHTML = '<span style="color: #10b981; font-weight: bold;">This email is classified as HAM (Not Spam)</span>';
  }
  if (data.spam_probability !== null) {
    const percentage = Math.round(data.spam_probability * 100);
    probabilityFill.style.width = percentage + '%';
    probabilityText.textContent = percentage + '% Spam Probability';
    probabilityBar.style.display = 'block';
  } else {
    probabilityBar.style.display = 'none';
  }
}

function useExample(text) {
  document.getElementById('email-text').value = text;
  showTab('text');
}