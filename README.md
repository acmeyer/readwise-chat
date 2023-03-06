# Readwise Chat

This is a repository for building a chatbot on top of your Readwise highlights. It's a bit experimental and there are a number of ways to potentially improve it but it's a start.

## How to use

Before doing anything else, you'll need to have a pro [Readwise account](https://readwise.io/) so that you can export your highlights via CSV.

Assuming you have an account, go to the [Exports Page](https://readwise.io/export) in your Readwise dashboard. From there, go to the CSV Export section and click the "Export" button. This will download a CSV file to your computer. Remember where this file is saved.

Next, install all the dependicies:

```
pip install -r requirements.txt
```

Then, create your own `.env` file by copying the `.env.example` file and filling in the values:

```
cp .env.example .env
```

Next, upload your CSV file you downloaded from Readwise by running:

```
python upload_data.py -f /path/to/your/csv/file
```

If this is your first time running this and you have a lot of highlights, this step could take awhile.

Once it's finished running, you can start the chatbot by running:

```
python qa.py
```

This will kick off a chat with the bot. Based on what you ask it, it should pull references to your highlights as you go, even updating those references based on where the conversation goes. However, it's not perfect and it may not always work the way you'd expect. 

For best results, try starting off the conversation asking it a specific question about a particular book and topic. The more specific you are, the better the results should be.

## How to improve

There are a few different ways you could improve the bot, including:
- Changing up the prompt, prompt engineering goes a long way
- Changing the strategy on pulling embeddings during the conversation, I haven't been able to figure out the best way to do this yet, where it feels like a natural conversation but the data for the bot continues to update itself well
- Giving the user options for what books it has at its disposal before starting the chat

And I'm sure there are a ton of other ways you could improve it. If you have any ideas, feel free to open an issue or a PR.