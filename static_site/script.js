document.getElementById('calcForm').addEventListener('submit', async function(event) {
  event.preventDefault();
  const userInput = document.getElementById('userInput').value;
  
  // Call the Python API (deployed elsewhere) and get the result
  const response = await fetch('/calculate', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input: userInput }),
  });
  
  const data = await response.json();
  document.getElementById('result').innerText = data.result;
});
