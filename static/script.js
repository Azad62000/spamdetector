function showTab(tabName) {
  const tabs = document.querySelectorAll('.tab-content');
  const buttons = document.querySelectorAll('.tab-button');
  tabs.forEach(tab => tab.classList.remove('active'));
  buttons.forEach(button => button.classList.remove('active'));
  document.getElementById(tabName + '-tab').classList.add('active');
  // Find the button that was clicked
  const clickedButton = Array.from(buttons).find(btn => btn.getAttribute('onclick').includes(tabName));
  if (clickedButton) clickedButton.classList.add('active');
}

function useExample(text) {
  document.getElementById('email-text').value = text;
  showTab('text');
}
