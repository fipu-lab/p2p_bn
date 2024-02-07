from models.util import save_json
from google.cloud import bigquery


def download_data(credentials_path, user_batch_size=100_000):

    # 20082092 -> comments
    # 20081043 -> questions
    # 20081009 -> answers

    client = bigquery.Client.from_service_account_json(credentials_path)
    # dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")
    for i, step in enumerate(range(1, 20081043, user_batch_size*2)):
        user_id_range = [step, step + user_batch_size - 1]
        print(user_id_range, end=' ')
        query = f"""
                  SELECT u_id, creation_date, text, tags, type
                  FROM (
                      -- select comments with tags from the post
                      SELECT cm.u_id, cm.creation_date, cm.text, pq.tags, "comment" as type
                      FROM (
                              SELECT a.parent_id as q_id, c.user_id as u_id, c.creation_date as creation_date, c.text as text
                              FROM `bigquery-public-data.stackoverflow.comments` as c
                              INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` as a ON (a.id = c.post_id)
                              WHERE c.user_id >= {user_id_range[0]} AND c.user_id <= {user_id_range[1]}
                              
                              UNION ALL 
                              
                              SELECT q.id as q_id, c.user_id as u_id, c.creation_date as creation_date, c.text as text
                              FROM `bigquery-public-data.stackoverflow.comments` as c
                              INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` as q ON (q.id = c.post_id)
                              WHERE c.user_id >= {user_id_range[0]} AND c.user_id <= {user_id_range[1]}
                          ) as cm
                      INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` as pq ON (pq.id = cm.q_id)
                          
                      UNION ALL
                      -- select answers with tags related to the post
                      SELECT pa.owner_user_id as u_id, pa.creation_date as creation_date, pa.body as text, pq.tags as tags, "answer" as type
                      FROM `bigquery-public-data.stackoverflow.posts_answers` as pa
                      LEFT OUTER JOIN `bigquery-public-data.stackoverflow.posts_questions` as pq ON pq.id = pa.parent_id
                      WHERE pa.owner_user_id >= {user_id_range[0]} and pa.owner_user_id <= {user_id_range[1]}
                      
                      UNION ALL
                      -- select posts
                      SELECT pq.owner_user_id as u_id, pq.creation_date as creation_date, pq.body as text, pq.tags as tags, "question" as type
                      FROM `bigquery-public-data.stackoverflow.posts_questions` as pq
                      WHERE pq.owner_user_id >= {user_id_range[0]} and pq.owner_user_id <= {user_id_range[1]}
                      )
                  ORDER BY u_id, creation_date
                  """

        result = client.query(query).result().to_dataframe()
        # print(result)
        result['creation_date'] = result['creation_date'].astype(str)

        users = {str(u_id): {
                             'text': list(result[result.u_id == u_id].text),
                             'tags': list(result[result.u_id == u_id].tags),
                             'type': list(result[result.u_id == u_id].type),
                             'creation_date': list(result[result.u_id == u_id].creation_date)
                             } for u_id in result.u_id.unique()}

        save_json(f'data/stackoverflow/users/stackoverflow_{i}.json', users)
        del result


if __name__ == '__main__':
    download_data(credentials_path='/Users/robert/Files/projects/python/p2p_sgd/data/stackoverflow/credentials.json')


