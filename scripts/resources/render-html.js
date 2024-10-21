const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {
  // Launch a headless browser using the system-installed Chromium
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/chromium-browser', // Path to your system-installed Chromium
    headless: true, // Ensure headless mode is enabled
  });

  const page = await browser.newPage();
  const weekParameter = process.argv[2] || 'week1'; // Default to 'week1' if no argument is given
  const url = `http://10.12.2.118/presentation_template.html?week=${weekParameter}`;

  await page.goto(url, { waitUntil: 'networkidle0' });
  const content = await page.content();
  fs.writeFileSync('rendered-page.html', content);

  console.log(`Rendered HTML saved to rendered-page.html with week=${weekParameter}`);
  await browser.close();
})();

