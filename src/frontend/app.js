document.addEventListener("DOMContentLoaded", function () {

    const homeButton = document.getElementById("homeButton");
    const recordsButton = document.getElementById("recordsButton");
    const homeContent = document.getElementById("homeContent");
    const recordsContent = document.getElementById("recordsContent");
    const recordsTable = document.getElementById("recordsTable").getElementsByTagName('tbody')[0];
    
    homeButton.addEventListener("click", () => {
        homeContent.style.display = "block";
        recordsContent.style.display = "none";

        homeButton.classList.add("active");
        recordsButton.classList.remove("active");
    });

    recordsButton.addEventListener("click", () => {
        homeContent.style.display = "none";
        recordsContent.style.display = "block";
        
        recordsButton.classList.add("active");
        homeButton.classList.remove("active");

                // You can make an API call here to retrieve records from the database
        // For this example, I'll use dummy records
        const dummyRecords = [
            { text: "I hate you!", score: 0.75 },
            { text: "Have an okay day", score: -0.5 },
            { text: "I don't know about that", score: 0.25 },
        ];

        // Clear existing table rows
        recordsTable.innerHTML = "";

        // Populate the table with the records
        dummyRecords.forEach((record) => {
            const row = recordsTable.insertRow();
            const cell1 = row.insertCell(0);
            const cell2 = row.insertCell(1);
            cell1.textContent = record.text;
            cell2.textContent = record.score.toFixed(2);
        });
    });

    const inputText = document.getElementById("inputText");
    const analyzeButton = document.getElementById("analyzeButton");
    const scoreElement = document.getElementById("score");
    const loadingBar = document.getElementById("loadingBar");

    analyzeButton.addEventListener("click", () => {
        const text = inputText.value;

        loadingBar.style.width = "100%";
        // Simulate sending the text to the backend for analysis.
        // Replace this with your actual backend API call.
        //const dummyScore = Math.random() * 2 - 1; // Generate a random score between -1 and 1.
        fetch("http://localhost:8080/divert_to_model", {
            method: "POST", // Use the appropriate method (GET, POST, etc.) for your API.
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: text }),
        })
            .then((response) => {
                if (response.ok) {
                    return response.json();
                } else {
                    throw new Error("API request failed");
                }
            })
            .then((data) => {
                const sentimentScore = data.score; // Assuming your API returns a 'score' property.
                // Update the score display.
                scoreElement.textContent = `Score: ${sentimentScore.toFixed(2)}`;
                // Change background color based on the score.
                const backgroundColor = getColorFromScore(sentimentScore);
                document.body.style.backgroundColor = backgroundColor;
            })
            .catch((error) => {
                console.error("API call error:", error);
            })
            .finally(() => {
                // Hide the loading bar
                loadingBar.style.width = "0";
            });
    });

    function getColorFromScore(score) {
        const hue = (1 - score) * 60;
        //const hue = (score + 1) * 60; // Map score to a hue between 0 (green) and 120 (red).
        return `hsl(${hue}, 70%, 70%)`; // Adjust saturation and lightness as needed.
    }
});