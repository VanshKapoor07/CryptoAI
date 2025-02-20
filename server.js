const express = require('express');
const path = require('path');
const next = require('next');

const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });

const PORT = process.env.PORT || 3000;

app.prepare().then(async() => {
    const server = express();
    // Middleware to parse URL-encoded bodies
    server.use(express.urlencoded({ extended: true }));

    // Middleware to parse JSON bodies (optional if you plan to send JSON)
    server.use(express.json());
    server.use(express.static('public'));

    // Set EJS as the view engine
    server.set('view engine', 'ejs');
    server.set('views', path.join(__dirname, 'pages', 'views')); // Adjust path if needed

    // Serve the EJS index page at the root URL
    server.get('/', (req, res) => {
        res.render('index'); // This should match your index.ejs file
    });

    server.listen(PORT, (err) => {
        if (err) throw err;
        console.log(`> Ready on http://localhost:${PORT}`);
      });
});