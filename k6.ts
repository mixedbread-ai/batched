import http from 'k6/http';
import { check } from 'k6';

export default function () {
  const url = 'http://localhost:8000/embeddings';
  
  const sampleTexts = [
    'This is a sample text for embedding.',
    'Another example of input for the model.',
    'Testing the embedding service with various inputs.',
    'Randomized text to check model performance.',
    'Different lengths and content for robust testing.'
  ];

  const randomText = sampleTexts[Math.floor(Math.random() * sampleTexts.length)];

  const randomChars = Math.random().toString(36).substring(2, 7);
  const inputText = `${randomText} ${randomChars}`;

  const payload = JSON.stringify({
    input: inputText,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const res = http.post(url, payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}
