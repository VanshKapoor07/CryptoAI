address 0x51e1de64d24251bbd3d9d96914e9cb8e79e08b3396c66c974d4b6087c0273916{
    module a5{
    use std::signer;
    use std::string;
    use std::vector;

    /// Error codes
    const E_NOT_INITIALIZED: u64 = 1;
    const E_ALREADY_INITIALIZED: u64 = 2;
    const E_INSUFFICIENT_BALANCE: u64 = 3;
    const E_TARGET_NOT_REACHED: u64 = 4;
    const E_NOT_AUTHORIZED: u64 = 5;


    /// Struct to hold the NFT data
    struct NFT has key,store, drop, copy  {
        id: u64,
        version: u64,
        owner: address,
        metadata_uri: string::String,
        type: string::String,
    }

    struct Spectator_data has key,store,drop,copy {
        id: u64,
        owner: address,
        nft_A: u64,
        nft_B: u64,
        nft_C: u64,
        resale_profit: u64,
    }

     /// Struct to hold the Organizer dashboard data
    struct Organizer_data has key,store, drop, copy  {
        id: u64,
        version: u64,
        owner: address,
        num_A: u64,
        num_B: u64,
        num_C: u64,
        unit_price_A: u64,
        unit_price_B: u64,
        unit_price_C: u64,
        price_A: u64,
        price_B: u64,
        price_C: u64,
        total_revenue: u64,
        resale_profit: u64,
    }


    

    public entry fun initialize_spectator_data(account: &signer){
        let address = signer::address_of(account);

        let spectator_data = Spectator_data {
            id: 5820582933,
            owner: address,
            nft_A: 0,
            nft_B: 0,
            nft_C: 0,
            resale_profit: 0,
        };
        move_to(account, spectator_data);
    }

    public entry fun initialize_organizer_data(account: &signer, unit_price_A: u64, unit_price_B: u64, unit_price_C: u64){
        let address = signer::address_of(account);

        let organizer_data = Organizer_data {
            id: 3284894984845,
        version: 1,
        owner: address,
        num_A: 0,
        num_B: 0,
        num_C: 0,
        unit_price_A: unit_price_A,
        unit_price_B: unit_price_B,
        unit_price_C: unit_price_C,
        price_A: 0,
        price_B: 0,
        price_C: 0,
        total_revenue: 0,
        resale_profit: 0,
        } ;
        move_to(account, organizer_data);
        
    }

    //  NFT Collection struct for storing multiple NFTs
    struct NFT_Collection has key, store {
        nfts: vector<NFT>, // Vector to store multiple NFTs
    }

        //  Initialize an empty NFT collection for the user
    public entry fun initialize_collection(account: &signer) {
        let address = signer::address_of(account);
        let collection = NFT_Collection { nfts: vector::empty<NFT>() };
        move_to(account, collection); // Store the empty collection in the user's account
    }


    
   
    #[view]
    public fun get_nft_data(nft_owner: address): (vector<NFT>) acquires NFT_Collection {
       
        let nft_data = borrow_global_mut<NFT_Collection>(nft_owner);
        (nft_data.nfts)
    }

    #[view]
    public fun get_organizer_data(owner: address): (u64,u64, address, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) acquires Organizer_data {
       
        let data = borrow_global_mut<Organizer_data>(owner);
        (data.id, data.version, data.owner, data.num_A, data.num_B, data.num_C, data.unit_price_A, data.unit_price_B, data.unit_price_C, data.price_A, data.price_B, data.price_C, data.total_revenue, data.resale_profit)

    }

    #[view]
    public fun get_spectator_data(spectator: address): (u64, address, u64, u64, u64, u64) acquires Spectator_data {
       
        let spectator_data = borrow_global_mut<Spectator_data>(spectator);
        (spectator_data.id,spectator_data.owner,spectator_data.nft_A,spectator_data.nft_B,spectator_data.nft_C, spectator_data.resale_profit)
    }


    
    public entry fun purchase_ticket(spectator: &signer, event_organizer: address, num_A: u64, num_B: u64, num_C: u64) acquires Organizer_data, Spectator_data, NFT_Collection {
        let spectator_address = signer::address_of(spectator);
        let organizer_data = borrow_global_mut<Organizer_data>(event_organizer);
        let spectator_data = borrow_global_mut<Spectator_data>(spectator_address);

        let price_A = organizer_data.unit_price_A * num_A;
        let price_B = organizer_data.unit_price_B * num_B;
        let price_C = organizer_data.unit_price_C * num_C;

        let total_price = price_A + price_B + price_C;
       
        organizer_data.num_A = organizer_data.num_A + num_A;
        organizer_data.num_B = organizer_data.num_B + num_B;
        organizer_data.num_C = organizer_data.num_C + num_C;
        
        organizer_data.price_A = organizer_data.price_A + (organizer_data.unit_price_A*num_A);
        organizer_data.price_B = organizer_data.price_B + (organizer_data.unit_price_B*num_B);
        organizer_data.price_C = organizer_data.price_C + (organizer_data.unit_price_C*num_C);
        organizer_data.total_revenue = organizer_data.total_revenue + total_price;


        spectator_data.nft_A = spectator_data.nft_A + num_A;
        spectator_data.nft_B = spectator_data.nft_B + num_B;
        spectator_data.nft_C = spectator_data.nft_C + num_C;

        let collection = borrow_global_mut<NFT_Collection>(spectator_address);
        

        let metadata_uri_A = string::utf8(b"https://emerald-voluntary-jay-831.mypinata.cloud/ipfs/bafkreib3xqkkg7larkhrf5zqk25dfaore26o5v4nugzwg3o3ewbni7v52i");
        let metadata_uri_B = string::utf8(b"https://emerald-voluntary-jay-831.mypinata.cloud/ipfs/bafkreicyevlt5cg3yqo4yiavricohnf2z22kgeuuws3qiozxzkezpebugy");
        let metadata_uri_C = string::utf8(b"https://emerald-voluntary-jay-831.mypinata.cloud/ipfs/bafkreigq53b4rw3pvl7flaoyelnzra47lt5u3l6uzya4b6xdnvpb62blvi");

        let type_A = string::utf8(b"Type- A (Premium)");
        let type_B = string::utf8(b"Type- B (Economy)");
        let type_C = string::utf8(b"Type- C (Price saving)");
        //  Generate NFTs for `A` category
        let i = 0;
        while (i < organizer_data.num_A) {
            let nft = NFT {
                id: 1000000000 + i,  // Unique ID generation
                version: 1,
                owner: spectator_address,
                metadata_uri: metadata_uri_A,
                type: type_A,
            };
            vector::push_back(&mut collection.nfts, nft);

            i = i + 1;
            
        };

        //  Generate NFTs for `B` category
        let j = 0;
        while (j < organizer_data.num_B) {
            let nft = NFT {
                id: 2000000000 + j,  // Unique ID for B
                version: 2,
                owner: spectator_address,
                metadata_uri: metadata_uri_B,
                type: type_B,
            };
            vector::push_back(&mut collection.nfts, nft);

            j = j + 1;
        };

        //  Generate NFTs for `C` category
        let k = 0;
        while (k < organizer_data.num_C) {
            let nft = NFT {
                id: 3000000000 + k,  // Unique ID for C
                version: 3,
                owner: spectator_address,
                metadata_uri: metadata_uri_C,
                type: type_C,
            };
            k = k + 1;
        vector::push_back(&mut collection.nfts, nft);

        };
        
        
    }


    /// Function to verify owner
    public entry fun verify_owner(signer: &signer, nft_id: u64) acquires NFT_Collection {
        let collection = borrow_global<NFT_Collection>(signer::address_of(signer));
        
        let sender_address = signer::address_of(signer);

        let found = false;
        // Iterate using while loop
        let i = 0;
        let nfts_ref = &collection.nfts;
        while (i < vector::length(nfts_ref)) {
            let nft = *vector::borrow(nfts_ref, i);
            if (nft.id == nft_id) {
                if (nft.owner == sender_address) {
                    found = true;
                    break;
                }
            };
            i = i + 1;
        };

        // If not owner, abort the transaction
        if (!found) {
            abort 5; // Custom error code
        }

        // If owner, transaction proceeds, no problems
    }

    public entry fun resale(signer: &signer, num_A: u64, num_B: u64, num_C: u64, prcnt_scaled: u64) acquires Spectator_data, Organizer_data {
        let spectator_data = borrow_global_mut<Spectator_data>(signer::address_of(signer));
        let organizer_data = borrow_global_mut<Organizer_data>(signer::address_of(signer));


        spectator_data.nft_A = spectator_data.nft_A - num_A;
        spectator_data.nft_B = spectator_data.nft_B - num_B;
        spectator_data.nft_C = spectator_data.nft_C - num_C;

        let spectator_profit = 4*(prcnt_scaled*(organizer_data.unit_price_A * num_A + organizer_data.unit_price_B * num_B + organizer_data.unit_price_C * num_C));
        
        let organizer_profit = 6*(prcnt_scaled*(organizer_data.unit_price_A * num_A + organizer_data.unit_price_B * num_B + organizer_data.unit_price_C * num_C));

        spectator_data.resale_profit = spectator_data.resale_profit + spectator_profit;
        organizer_data.resale_profit = organizer_data.resale_profit + organizer_profit;

    }

}
}

