const axios = require("axios");

async function getNFTPrice(metadataUri, name) {
    try {
        const response = await axios.get(metadataUri);
        const data = response.data; // NFT Metadata JSON

        let price = null;

        // Search for "Price" attribute
        for (const attr of data.attributes) {
            if (attr.trait_type === "Level") {
                price = attr.value;
                break;
            }
        }
        
       
        if (price !== null) {
            return price;
        } else {
            console.log("Price not available in metadata.");
        }

    } catch (error) {
        console.error("Error fetching metadata:", error.message);
    }
}

async function fetchPrices() {
    const metadataUriA = "https://emerald-voluntary-jay-831.mypinata.cloud/ipfs/bafkreib3xqkkg7larkhrf5zqk25dfaore26o5v4nugzwg3o3ewbni7v52i";
    const metadataUriB = "https://emerald-voluntary-jay-831.mypinata.cloud/ipfs/bafkreicyevlt5cg3yqo4yiavricohnf2z22kgeuuws3qiozxzkezpebugy";
    const metadataUriC = "https://emerald-voluntary-jay-831.mypinata.cloud/ipfs/bafkreigq53b4rw3pvl7flaoyelnzra47lt5u3l6uzya4b6xdnvpb62blvi";

    const priceA = await getNFTPrice(metadataUriA, "A");
    const priceB = await getNFTPrice(metadataUriB, "B");
    const priceC = await getNFTPrice(metadataUriC, "C");

    console.log(`Price A: ${priceA}`);
    console.log(`Price B: ${priceB}`);
    console.log(`Price C: ${priceC}`);
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve({
                priceA: priceA,
                priceB: priceB,
                priceC: priceC
            });
        }, 1000);
    });
}

// Call the function
fetchPrices();
module.exports = fetchPrices;

