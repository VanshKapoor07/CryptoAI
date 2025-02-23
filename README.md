Problem Statement - 
The problem was to implement a blockchain based decentralized ticket booking and event registering platform.

Our proposed solution - 
Our solution (CryptoAI) serves as a simple , safe , smart(powered by AI) and user freindly solution to the problem.

-->What really a ticket is here?
  -->Each ticket(NFT) represents a seat in the event .
  -->Which futher goes to a verification process using its unique id.
  
-->Registering an event and ticket booking
  --> We have developed an Event organizer and spectator's interface where anyone can register to register for ticket 
      selling of his/her event(filling up the relevant details) and decide what the ticket prices for the seats as per 
      his/her choice.
  --> All the events are displayed on user's dashboard sorted according to the genre(for eg - music,education,comedy etc) 
      and clicking on one would redirect the user to ticket booking platform where one can select tickets based on the 
      pricepoint(platinum or gold or silver) and availabilty.
  -->The total number of tickets one can book is restricted to 5(to eliminate scalping).
  -->This all is done by sale* function implemented in the smart contract.

-->Selling the ticket back
  -->If someone wishes to sell the ticket back to someone because of any reason it can be done under under some conditions only 
     1)Profit (extra price over the original) can be set between 0 and 20 percent only.
     2)60% of the total profit goes to the organiser and 40% to the previous owner.
  -->This all happens using resale function in the smart contract.
  -->In this way the NFT goes to the new owner from previous one.

-->Different time different price!!
  -->As the aptos or any crytocurrencies prices may change significantly within 5-6 hours it is important for the user to 
     know what is the best time to buy the NFT (the time where he/she has to pay least amount of USD).
  -->To serve the purpose we implemented LSTM algorithm (a RNN) which predicts prices of aptos currency for next 5 hours by 
     fetching past few hour data dynammically and passing it through a saved trained model(the RNN).
  -->The possible profit/loss(per apt) is displayed to the user for each of next 5 subsequent hour for him/her to decide the 
    optimal time to buy the ticket.
  -->The LSTM works for next few hours prediction whereas the sentiment analysis model (fine tuned GPT-02) gives behaviour 
     of the trend depicting whether market is predicted to be bullish/bearish/sidish.

-->Knowing and comparing events and hosts
  -->One might be in a mood of attending a comedy show but might not be aware of the top artists based on certain parameters 
     relevant for him/her.
  -->Our RAG model comes to the rescue here where it uses nlp and text vectorisation to answer user queries based on the 
     data given by the host given at time of registration or retrives from the web.
