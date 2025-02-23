const express = require('express');
const next = require('next');
const path = require('path');
const mysql = require('mysql2');
const multer = require('multer');
const storage = multer.memoryStorage();
const upload = multer({ storage: storage ,limits: {
    fileSize: 5 * 1024 * 1024 // 5MB limit
}});

const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });
const PORT = process.env.PORT || 8000;
//const app=express();
var flag=false;

const connection = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'monkey_#lindo@9731',
    database: 'users',
});

connection.connect((err) => {
    if (err) throw err;
    console.log('Connected to MySQL database');
});

app.prepare().then(() => {
    const server = express();
    
    server.use(express.urlencoded({ extended: true }));
    server.use(express.json());
    server.use(express.static('public'));

    server.set('view engine', 'ejs');
    server.set('views', path.join(__dirname, 'pages', 'views')); 

    // Serve the main page
    server.get('/', (req, res) => {
        const sql = `
            SELECT events.*, 
                   user_images.image 
            FROM events 
            LEFT JOIN user_images ON events.Artist = user_images.username`;
    
        connection.query(sql, (err, results) => {
            if (err) {
                console.error(err);
                return res.status(500).send("Database query failed");
            }
    
            const bookings = results.map(event => ({
                title: event.Event_Name,
                date: event.Date,
                description: event.About_event,
                image: event.image 
                    ? `data:image/jpeg;base64,${event.image.toString('base64')}`
                    : "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Unknown_person.jpg/694px-Unknown_person.jpg",
                Venue: event.Venue,
                Time: `${event.Start_Time} - ${event.End_Time}`,
                category: event.Event_Category,
                Artist: event.Artist
            }));
    
            res.render("index", { bookings });
        });
    });
    
    
    server.get('/RAG', (req, res) => {
        res.render('RAG'); 
    });

    server.get('/lstm', (req, res) => {
        res.render('lstm'); 
    });

   let message=false;
    server.get('/sign_in', (req, res) => {
        res.render('sign_in',{ message}); 
    });


    server.post('/signup', (req, res) => {
        const { name, email, password, confirmpassword } = req.body;
    
        if (!name || !password || !confirmpassword || !email) {
            return res.status(400).send("Invalid Input");
        }
        if (password === confirmpassword) {
            connection.query('INSERT INTO userInfo(User_Name, Email_Id, Password, Confirm_Password) values(?, ?, ?, ?)', 
                [name, email, password, confirmpassword], (err, result) => {
                    if (err) {
                        console.error(err);
                        return res.status(500).send('Error inserting user');
                    }
                    res.render('profile', { name });
                });
        }
    });


    server.post('/login', (req, res) => {
        const { name, password } = req.body;
        if (!name || !password) {
            return res.status(400).send("Invalid Input");
        }
    
        if (connection.state === 'disconnected') {
            connection.connect(err => {
                if (err) {
                    console.error('Error connecting to MySQL:', err);
                    return res.status(500).send('Database connection error');
                }
            });
        }
    
        connection.query('SELECT * FROM userInfo WHERE User_Name = (?) AND Password = (?)', [name, password], (err, results) => {
            if (err) {
                console.error(err);
                return res.status(500).send('Error executing query');
            }
    
            if (results.length > 0) {
                const userName = results[0].User_Name;
                const passWord=results[0].Password;
                console.log(results);
                res.render('profile', { name: userName ,password:passWord});
            } else {
                message = "Wrong credentials, try again";
                res.render('index', { message });
            }
        });
    });

    server.post('/event_api',(req,res)=>{
        const { artist_name,Event_Name, Date, Event_Category,About_Event, Venue,Start_Time,End_Time,plat_price,gold_price,silver_price } = req.body;
    
            connection.query('INSERT INTO events(Artist, Event_Name, Date,Venue,About_event,Event_Category,Start_Time,End_Time,Platinum_price,Gold_price,Silver_price) values(?,?,?,?,?,?,?,?,?,?,?)', 
                [artist_name,Event_Name, Date, Venue,About_Event,Event_Category,Start_Time,End_Time,plat_price,gold_price,silver_price], (err, result) => {
                    if (err) {
                        console.error(err);
                        return res.status(500).send('Error inserting user');
                    }
                    res.render('profile');
                });
        

    });

    server.post('/upload', upload.single('image'), (req, res) => {
        try {
            // Check if file exists
            if (!req.file) {
                return res.status(400).send('No file uploaded');
            }
    
            const username = req.body.username;
            const filename = req.file.originalname;
            const image = req.file.buffer;
    
            connection.query('INSERT INTO user_images (username, filename,image) VALUES (?, ?,?)', [username,filename, image], (err, result) => {
                if (err) {
                    console.error('Database error:', err);
                    return res.status(500).send('Error saving to database');
                }
                res.redirect('/');
            });
        } catch (error) {
            console.error('Upload error:', error);
            res.status(500).send('Server error during upload');
        }
    });

    server.listen(PORT, (err) => {
        if (err) throw err;
        console.log(`> Ready on http://localhost:${PORT}`);
    });
});
