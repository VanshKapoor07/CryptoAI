//----------------------------------------------------------------------------------------------------------------------------------//
// Part 1 --> Using express to render EJS
const express = require('express');
const app = express();
const path = require('path');
const axios = require('axios');


// Serve static files
app.use(express.static(path.join(__dirname, 'views')));

app.use(express.json()); // Middleware to parse JSON bodies
app.set('view engine', 'ejs');
app.use(express.urlencoded({ extended: false }));

let message = false;

// Render EJS pages
app.get('/', (req, res) => {
    res.render('index', { message });
});

//----------------------------------------------------------------------------------------------------------------------------------//
// Part 2 --> Fetch latest news using CryptoPanic API

async function getCryptoNews(page, currency) {
    try {
        const response = await axios.get('https://cryptopanic.com/api/v1/posts/', {
            params: {
                currencies: currency,
                auth_token: '04b5353affcbe5e7e4dee00684c5a27b5325ea1a',
                page: page,
            },
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching news:', error.message);
    }
}

async function getMultiplePagesNews(totalPages, currency) {
    let newsData = []; // Array to store all news
    for (let i = 1; i <= totalPages; i++) {
        const pageNews = await getCryptoNews(i, currency);
        newsData = newsData.concat(pageNews); // Append news from this page
    }
    return newsData;
}

//----------------------------------------------------------------------------------------------------------------------------------//
// Part 3 --> Send news to app.py and display it on EJS page

async function sendNewsToPython(newsData) {
    try {
        const response = await axios.post('http://127.0.0.1:5000/receive_news', {
            news: newsData
        });
        console.log("Sent news to Python:", response.data);
        return response.data; // Return analyzed sentiment data
    } catch (error) {
        console.error("Error sending news to Python:", error.message);
        return [];
    }
}

app.post('/shownews', async (req, res) => {
    let newsDataapt = await getMultiplePagesNews(10, "APT");
    let allNews = [];

    for (let i = 0; i < 10; i++) {
        let resultsapt = newsDataapt[i]["results"];
        let l = resultsapt.length;
        
        for (let j = 0; j < l; j++) {
            let titleapt = resultsapt[j]["title"];
            let dateapt = resultsapt[j]["published_at"];

            allNews.push({ title: titleapt, date: dateapt });
        }
    }

    // Send news to Python for sentiment analysis
    let analyzedNews = await sendNewsToPython(allNews);

    res.render('shownews', {
        analyzedNews: analyzedNews // Pass sentiment-analyzed news to EJS
    });
});

//----------------------------------------------------------------------------------------------------------------------------------//
// Part 4 --> Run the server on port 3005

const PORT = process.env.PORT || 3005;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
