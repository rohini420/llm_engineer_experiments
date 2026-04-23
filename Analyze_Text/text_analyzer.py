from openai import OpenAI

API_KEY = "xxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
client = OpenAI(
  api_key=API_KEY,
)

TEXT_TO_ANALYSE = "I find the software very intuitive and user-friendly"

def analyze_sentiment(text):
  
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Analyze the sentiment of the following text: \"{text}\". Is it positive, negative, or neutral? Answer in one word with no punctuation."}
        ],
        max_tokens=50
    )
  sentiment = response.choices[0].message.content.strip().lower()
  if sentiment not in ["positive", "negative", "neutral"]:
    return "unable to determine sentiment. please try again later"
    
  return f"The sentiment of the text is: {sentiment}"
    
result = analyze_sentiment(TEXT_TO_ANALYSE)
print(result)

# Testing the analyzer
# Now that the program is done, use the following sentences to test how well the sentiment analysis program performs.

# "The sun is shining, the birds are singing, and it's a beautiful day!"

# "I am devastated to hear about the tragic event that took place last night."

# "The movie's plot was quite mediocre, and the character development was lackluster."

# "I've never tasted ice cream this delicious before!"

# "The meeting was long and tedious, but it was necessary to get everyone on the same page."

# "It's just a regular day; nothing special happened."

# "Reading that book was a transformational experience for me."

# "The concert was an absolute disaster; the band was off-key, and the sound system kept malfunctioning."

# "I find the software very intuitive and user-friendly."

# "It's a confusing process, and the instructions provided don't help at all."