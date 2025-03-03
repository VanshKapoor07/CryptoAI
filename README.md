# CryptoAI - Blockchain-based Decentralized Ticket Booking and Event Registration Platform

## 1. How to Run?

### (a) LSTM
1. Open the IPYNB file and execute each block to train the model.
2. Run the following command to see the LSTM prediction results:
   ```bash
   python app_1.py
   ```

### (b) RAG
1. Install the required dependencies like `langchain`.
2. Run the following command:
   ```bash
   python app.py
   ```

### (c) Finetuned LLM
1. Download the required files:
   - **Eth_predictions.pth**: [Download Here](https://drive.google.com/file/d/1nPT45tri3Sl6kdNXJRj6DrqPlN8GpK1Z/view?usp=drive_link)
   - **download_gpt_3.py**: Available on the internet.
2. After downloading, run the following command to start the application:
   ```bash
   python app.py
   ```
   The finetuned model will load and give predictions.

### (d) Blockchain (Main Project)
1. Run the backend server:
   ```bash
   node server.js
   ```
2. Deploy the smart contract and initialize it at address: `0x51e1de64d24251bbd3d9d96914e9cb8e79e08b3396c66c974d4b6087c0273916`.
3. Call the following functions with the required attributes:
   - `initialize_collection`
   - `initialize_spectator_data`
4. Run the backend server again:
   ```bash
   node server.js
   ```
5. Open the browser and interact with the website normally to see the results.

---

## 2. Repository Structure

```
CryptoAI/
│── .next/                        # Generated files (Next.js build output)
│── KrackHack 2.O Move/aptos/KrackHack 2.O/                        # Move code
│   ├── out                      
│   │   ├── a5.mv                     # config file
│   │   ├── package-metadata.bcs      # config file
│   ├── sources
│   │   ├── main.move                 # Main smart contract implementation in Move
│   ├── Move.toml                  #toml file
│── more_than_hour/               # (Unclear purpose)
│── pages/views/                  # Views for front-end rendering
│   ├── Booking.ejs
│   ├── RAG.ejs
│   ├── consumer.ejs              # Consumer and service provider login pages
│   ├── index.ejs
│   ├── index1.ejs
│   ├── initialize_organizer_data.ejs
│   ├── profile.ejs
│   ├── purchase_ticket.ejs
│   ├── purchase_ticket_payment.ejs
│   ├── resale.ejs
│   ├── service_provider.ejs
│   ├── sign_in.ejs
│   ├── verification.ejs
│── rag/                          # Retrieval-Augmented Generation (RAG) related files
│   ├── templates/                 # Templates for RAG
│   │   ├── RAG.ejs
│   ├── .env
│   ├── app.py                     # Python application script
│── README.md                     # Project documentation
│── app2.py                        # Another Python application script
│── get_price.js                   # Fetch cryptocurrency prices
│── main2.js                        # Main JavaScript script
│── metadataA.json                  # Metadata files
│── metadataB.json
│── metadataC.json
│── server.js                       # Server-side JavaScript file
│── server2.js                      # Another server script
```

---

## 3. Brief Summary of the Code

The code provides a comprehensive way to perform transactions among people offering various types of digital services related to entertainment, music, education, etc.

### Features Implemented via Smart Contract:
- **Ticket Booking Management**
  - Restricts maximum seats booked per user to prevent black marketing.
  - Maps unique token IDs to users for validation, eliminating the need for physical tickets or QR codes.
- **Ticket Resale System**
  - Limits the resale profit to a maximum of **20%**.
  - **60% of the profit** goes to the event organizer, and **40%** to the seller.
- **Digital Verification Process**
  - Ensures security and authenticity via smart contracts.
- **NFT Creation**
  - Generates digital NFTs with unique IDs mapped to owner addresses.
- **AI-powered Predictions**
  - **LSTM Model** predicts **Aptos cryptocurrency prices** based on historical data.
  - **RAG Model** answers user queries based on event details and web data.
  - **Finetuned GPT-3 Model** performs sentiment analysis for better stock prediction.

---

## 4. Detailed Description

### Problem Statement
The problem was to implement a blockchain-based **decentralized ticket booking and event registration platform**.

### Proposed Solution - **CryptoAI**
CryptoAI serves as a **simple, safe, AI-powered, and user-friendly** solution to the problem.

### Ticketing System
- Each **ticket (NFT)** represents a **seat in the event**.
- The ticket undergoes a **verification process** using its **unique ID**.

### Event Registration & Ticket Booking
- **Event Organizers** can register their events, set ticket prices, and list them.
- Events are categorized by genre (**music, education, comedy, etc.**).
- Users can book tickets based on price categories (**Platinum, Gold, Silver**) and availability.
- A single user can **book up to 5 tickets** to prevent scalping.
- The `sale` function in the smart contract **handles ticket sales**.

### Ticket Resale System
- Users can resell their tickets **under specific conditions**:
  1. **Profit cap**: Additional resale price is limited to **20%** over the original price.
  2. **Profit sharing**: **60%** of resale profit goes to the organizer, **40%** to the seller.
- The `resale` function in the smart contract **handles NFT ownership transfer**.

### AI-powered Pricing Insights
- **Cryptocurrency price fluctuations** impact ticket prices.
- To help users find the best time to buy tickets, we use **LSTM (RNN) models**:
  - Predicts **Aptos cryptocurrency prices** for the **next 5 hours**.
  - Fetches live historical price data dynamically.
  - Displays **expected profit/loss per APT token** to help users decide the **best purchase time**.
  - LSTM works for short-term predictions, while **sentiment analysis (fine-tuned GPT-3)** determines **bullish/bearish/sidish market trends**.

### Event & Host Comparison via RAG Model
- Users may want to **compare events** or **find the best artists**.
- Our **RAG model**:
  - Uses **NLP and text vectorization** to answer user queries.
  - Fetches data from **event registrations and the web** to provide insights.

---

## Conclusion
CryptoAI **combines blockchain, AI, and smart contracts** to create a **transparent, efficient, and secure** event ticketing system. It eliminates ticket scalping, ensures authenticity, and leverages AI to optimize ticket purchasing based on cryptocurrency price trends. The platform makes ticketing more accessible, secure, and intelligent.

