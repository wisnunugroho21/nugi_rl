from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.test_db
users = db.users
users.remove()
users.insert_one({
    'name': 'Keren'
})
#print(users.find_one())
for x in users.find():
    print(x)