const express = require('express');
const path = require('path');
const next = require('next');
const mysql = require('mysql2');
const multer = require('multer');
const storage = multer.memoryStorage();
const upload = multer({ storage: storage ,limits: {
    fileSize: 5 * 1024 * 1024 // 5MB limit
}});


const { Aptos, AptosConfig, Network } = require("@aptos-labs/ts-sdk");

const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });

const PORT = process.env.PORT || 3000;

const connection = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'your_password',
    database: 'users',
});

connection.connect((err) => {
    if (err) throw err;
    console.log('Connected to MySQL database');
});

const config = new AptosConfig({ network: Network.TESTNET });
const client = new Aptos(config);

const fetchPrices = require("./get_price");
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
    
            res.render("index1", { bookings });
        });
    });


    server.post('/service_provider',(req, res) =>{
        res.render('service_provider');
        
    })
    let message = false;
    server.get('/sign_in', (req, res) => {
        res.render('sign_in',{ message}); 
    });


    // async function getImageBase64(url) {
    //     const response = await axios.get(url, { responseType: 'arraybuffer' });
    //     return `data:image/jpeg;base64,${Buffer.from(response.data, 'binary').toString('base64')}`;
    // }

    server.post('/signup', async (req, res) => {
        //let imageBase64 = await getImageBase64("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Unknown_person.jpg/694px-Unknown_person.jpg");

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
                
        const num_A = 0;
        const num_B = 0;
        const num_C = 0;
        const unit_price_A = 0;
        const unit_price_B = 0;
        const unit_price_C = 0;
        const name2 = null;
        const num_A2 = 0;
        const num_B2 = 0;
        const num_C2 = 0;

        const total_revenue = num_A*unit_price_A + num_B*unit_price_B + num_C*unit_price_C;
        res.render('profile',{name:name, unit_price_A:unit_price_A,unit_price_B:unit_price_B,unit_price_C:unit_price_C,num_A:num_A,num_B:num_B,num_C:num_C, total_revenue:total_revenue,name2:name2, num_A2:num_A2, num_B2:num_B2, num_C2:num_C2}); 
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
                // connection.query('SELECT image FROM user_images WHERE username = ?', [userName], (err, imageResults) => {
                //     if (err) {
                //         console.error(err);
                //         return res.status(500).send('Error fetching user image');
                //     }
    
                    // let imageBase64 = null;
                    // if (imageResults.length > 0) {
                    //     imageBase64 = imageResults[0].image.toString('base64'); // Convert image to base64
                    // }
                    const num_A = 0;
        const num_B = 0;
        const num_C = 0;
        const unit_price_A = 0;
        const unit_price_B = 0;
        const unit_price_C = 0;
        const name2 = null;
        const num_A2 = 0;
        const num_B2 = 0;
        const num_C2 = 0;
        const total_revenue = num_A*unit_price_A + num_B*unit_price_B + num_C*unit_price_C;
        res.render('profile',{name:name, unit_price_A:unit_price_A,unit_price_B:unit_price_B,unit_price_C:unit_price_C,num_A:num_A,num_B:num_B,num_C:num_C, total_revenue:total_revenue,name2:name2, num_A2:num_A2, num_B2:num_B2, num_C2:num_C2}); 
// });
            } 
            else {
                message = "Wrong credentials, try again";
                res.render('index', { message });
            }
        });
    });



    server.post('/consumer',(req,res)=>{
        res.render('consumer');
})
    server.post('/purchase_ticket',(req,res)=>{
        res.render('Booking');
    })

    server.post('/purchase_all', async(req,res)=>{
        try {
            //const prices = await fetchPrices();
            const { num_A, num_B, num_C } = req.body;
            console.log("Ready now");
            console.log(num_A);
            console.log(num_B);
            console.log(num_C);
            
            res.render('purchase_ticket_payment',{client: client, num_A:num_A, num_B:num_B, num_C:num_C});
        } catch (error) {
            res.status(500).json({ error: "Failed to fetch prices" });
        }
    })


    server.post('/initialize_organizer_data', (req,res)=>{
        const { name, Event_Name, Date, Event_Category, About_Event, Venue, Start_Time, End_Time, unit_price_A, unit_price_B, unit_price_C } = req.body;
        connection.query('INSERT INTO events(Artist, Event_Name, Date,Venue,About_event,Event_Category,Start_Time,End_Time,Platinum_price,Gold_price,Silver_price) values(?,?,?,?,?,?,?,?,?,?,?)', 
                [name,Event_Name, Date, Venue,About_Event,Event_Category,Start_Time,End_Time,unit_price_A,unit_price_B,unit_price_C], (err, result) => {
                    if (err) {
                        console.error(err);
                        return res.status(500).send('Error inserting user');
                    }
            });
        console.log(name,unit_price_A,unit_price_B,unit_price_C);
        console.log("done");
        res.render('initialize_organizer_data',{name: name, unit_price_A: unit_price_A, unit_price_B: unit_price_B, unit_price_C: unit_price_C, client : client})
        
    })

    server.post('/back_to_profile',(req,res)=>{
        const name = "Sahil_SQL";
        const num_A = 2;
        const num_B = 3;
        const num_C = 4;
        const unit_price_A = 50;
        const unit_price_B = 40;
        const unit_price_C = 10;
        const name2 = null;
        const num_A2 = null;
        const num_B2 = null;
        const num_C2 = null;
        const total_revenue = num_A*unit_price_A + num_B*unit_price_B + num_C*unit_price_C;
        res.render('profile',{name:name, unit_price_A:unit_price_A,unit_price_B:unit_price_B,unit_price_C:unit_price_C,num_A:num_A,num_B:num_B,num_C:num_C, total_revenue:total_revenue,name2:name2, num_A2:num_A2, num_B2:num_B2, num_C2:num_C2});   
     })

    server.post('/back_to_profile_post_purchase',(req,res)=>{
        const name = "Sahil_SQL";
        const num_A = 2;
        const num_B = 3;
        const num_C = 4;
        const unit_price_A = 50;
        const unit_price_B = 40;
        const unit_price_C = 10;
        const name2 = "Vansh";
        const num_A2 = 1;
        const num_B2 = 2;
        const num_C2 = 3;
        const total_revenue = num_A*unit_price_A + num_B*unit_price_B + num_C*unit_price_C;

        res.render('profile',{name:name, unit_price_A:unit_price_A,unit_price_B:unit_price_B,unit_price_C:unit_price_C,num_A:num_A,num_B:num_B,num_C:num_C, total_revenue:total_revenue,name2:name2, num_A2:num_A2, num_B2:num_B2, num_C2:num_C2});   

    })

    server.post('/verify', (req,res)=>{
        const id = req.body["name"];
        res.render('verification',{id:id,client: client});
    })


    server.post('/resale', (req,res)=>{
        const num_A = req.body["A"];
        const num_B = req.body["B"];
        const num_C = req.body["C"];
        const prcnt = (req.body["prcnt"] / 10)
        res.render('resale',{num_A:num_A,num_B:num_B,num_C:num_C,prcnt:prcnt,client: client})
    })


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