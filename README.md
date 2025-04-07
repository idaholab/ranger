# moose-discussion-bot

[![GitHub license](https://img.shields.io/github/license/MengnanLi91/moose-discussion-bot)](https://github.com/MengnanLi91/moose-discussion-bot/blob/main/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/MengnanLi91/moose-discussion-bot)](https://github.com/MengnanLi91/moose-discussion-bot/graphs/contributors)

This repository contains a Python-based auto-response bot that leverages the GitHub API and the open-source LlamaIndex package to automatically generate responses to discussions on GitHub. The bot monitors GitHub MOOSE repository discussions and provides the most relevant posts when a new discussion is initiated by users.

## How It Works
The bot uses the GitHub API to fetch discussions from a MOOSE repository and store the data in a vector database. When a new discussion is initiated, the algorithm compares the discussion title with the content of all previous discussions (title + discussions) in the database and provides the most relevant posts to the user. The database is updated regularly to include all new posts, potentially on a monthly basis.

## Repository Contents
The repository includes the source code for discussion data parsing, vector database generation, relevant post suggestions, and unit test scripts.

### GitHubAPI.py
This script fetches relevant information from the GitHub discussion forum using GraphQL queries.

**Environment Variable Required:**
- `GITHUB_TOKEN`: The token should be granted sufficient access to read the repository.

**Functionality:**
- Automatically traverses through pagination to fetch information from each discussion post.
- Stores each page in JSON format, including the original question and comments.

**Prerequisites:**
- `query.gql.in`
- `GITHUB_TOKEN`

### IndexGenerator.py
This script embeds the relevant discussion information into a vector database using LlamaIndex functions.

**Functionality:**
- Uses `SimpleDirectoryReader` to read JSON data from `GitHubAPI.py` and save it as a `Document` object.
- Uses `HuggingFaceEmbedding` to load the embedding model. The default model is "all-MiniLM-L6-v2".
- Uses `SemanticSplitterNodeParser` to chunk content into nodes according to their semantic similarity.
- Uses `VectorStoreIndex` to generate the vector database and save it locally.

**Prerequisites:**
- Transformer model (default: all-MiniLM-L6-v2)
- JSON format discussion data (`\rawdata`)

### GitHubBot.py
This script loads the vector database, generates the most relevant posts according to the title of a new post, and posts the result as a reply.

**Functionality:**
- Uses cosine similarity search to find the similarity between the new post's title and previous posts' titles and discussion contents.
- Adjustable parameters: `top_n` (number of most relevant posts) and `threshold` (similarity cutoff).

**Prerequisites:**
- Transformer model (default: all-MiniLM-L6-v2)
- Vector database (`\db_dir`)

**Note:** It is recommended to use the same transformer model for vector database embedding and retrieval for best performance.

## Submodule
- `all-MiniLM-L6-v2` is a sentence-transformer model used to embed content into a vector index. It maps sentences and paragraphs to a 384-dimensional dense vector space for tasks like clustering or semantic search.

## Installation
To install and set up the `moose-discussion-bot`, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/idaholab/moose-discussion-bot.git
    ```
2. Navigate to the repository directory:
    ```bash
    cd moose-discussion-bot
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Configure the bot by creating a `.env` file with the necessary environment variables:
    ```plaintext
    GITHUB_TOKEN=your_github_token
    REPO_OWNER=your_repo_owner
    REPO_NAME=your_repo_name
    ```

## Testing
Separate unit tests are developed for each class in the repository using `unittest`. The tests are organized as follows:

1. `test_GitHubAPI.py`: Contains unit tests for `GitHubAPI.py`.
2. `test_IndexGenerator.py`: Contains unit tests for `IndexGenerator.py`.
3. `test_GitHubBot.py`: Contains unit tests for `GitHubBot.py`. The `testdatabase` in the repository is needed to complete the `test_load_database` unit test.

## Contact
If you have any questions or suggestions, feel free to contact @MengnanLi91.

## References
1. [MOOSE GitHub Mining](https://github.com/hugary1995/moose-gh-mining)
2. [Discuss-Bot](https://github.com/dhrubasaha08/Discuss-Bot)
