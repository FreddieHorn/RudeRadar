document.addEventListener("DOMContentLoaded", function () {
    const inputText = document.getElementById("inputText");
    const analyzeButton = document.getElementById("analyzeButton");
    const scoreElement = document.getElementById("score");

    analyzeButton.addEventListener("click", () => {
        const text = inputText.value;

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
            });
    });

    function getColorFromScore(score) {
        const hue = (1 - score) * 60;
        //const hue = (score + 1) * 60; // Map score to a hue between 0 (green) and 120 (red).
        return `hsl(${hue}, 70%, 70%)`; // Adjust saturation and lightness as needed.
    }
});