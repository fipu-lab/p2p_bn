from models.util import save_json
from google.cloud import bigquery


def download_data(credentials_path):
    client = bigquery.Client.from_service_account_json(credentials_path)
    # dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")
    query = """
              SELECT c.user_id as u_id, c.creation_date as creation_date, c.text as comment
              FROM `bigquery-public-data.stackoverflow.comments` as c
              WHERE c.user_id > 0
              ORDER BY u_id, creation_date
              LIMIT 5000000
            """
    result = client.query(query).result().to_dataframe()
    users = {str(u_id): list(result[result.u_id == u_id].comment) for u_id in result.u_id.unique()}
    save_json('data/stackoverflow/stackoverflow_0.json', users)


if __name__ == '__main__':
    download_data('/Users/robert/Files/projects/python/p2p_sgd/data/stackoverflow/credentials.json')


