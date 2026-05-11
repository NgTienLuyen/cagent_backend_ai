import math
import uuid
import tiktoken
import re
import json

def process_batch_to_postgres(batch_df, model, connection):
    """Encode and save the batch data to PostgreSQL in batches."""
    try:
        # Encode column data to vectors for this batch
        embeddings = model.encode(batch_df['chunk'].tolist())

        # Collect all metadata in one list (including the newly added '_id' column)
        metadatas = [row.to_dict() for _, row in batch_df.iterrows()]

        # Create a cursor to interact with PostgreSQL
        cursor = connection.cursor()

        # Prepare SQL query for inserting data
        insert_query = """
        INSERT INTO embeddings (id, chunk, embedding, metadata)
        VALUES (%s, %s, %s, %s)
        """

        # Insert each row into the database
        for idx, row in batch_df.iterrows():
            # Generate a unique id for the batch
            batch_id = str(uuid.uuid4())
            chunk = row['chunk']
            embedding = embeddings[idx].tolist()  # Convert the numpy array to a list
            metadata = json.dumps(metadatas[idx])  # Convert metadata dictionary to JSON

            # Execute the insert query
            cursor.execute(insert_query, (batch_id, chunk, embedding, metadata))

        # Commit the transaction to save the data
        connection.commit()
        cursor.close()

    except Exception as e:
        connection.rollback()  # Rollback in case of error
        print(f"Error saving data to PostgreSQL: {str(e)}")
        raise


def divide_dataframe(df, batch_size):
    """Divide DataFrame into smaller chunks based on the chunk size."""
    num_batches = math.ceil(len(df) / batch_size)
    return [df.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


# Count the number of tokens in each page_content
def openai_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def clean_collection_name(name):
    # Clean the name based on the required pattern
    # Allow only alphanumeric, underscores, hyphens, and single periods in between
    cleaned_name = re.sub(r'[^a-zA-Z0-9_.-]', '', name)  # Step 1: Remove invalid characters
    cleaned_name = re.sub(r'\.{2,}', '.', cleaned_name)  # Step 2: Remove consecutive periods
    cleaned_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '',
                          cleaned_name)  # Step 3: Remove leading/trailing non-alphanumeric characters

    # Ensure the cleaned name meets length constraints
    return cleaned_name[:63] if 3 <= len(cleaned_name) <= 63 else None


