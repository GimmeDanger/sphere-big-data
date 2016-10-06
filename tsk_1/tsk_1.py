import pandas as pd
import numpy as np
import pylab as pl
import mpl_toolkits.basemap as bm
import twitter
import requests
import datetime
import dateutil
import csv
import os
import json
import xml.etree.ElementTree as ET

# Part 1 -- Read dataset
TRAINING_SET_PATH = "train.csv"
TESTING_SET_PATH = "test.csv"
df_users_train = pd.read_csv(TRAINING_SET_PATH, sep=",", header=0)
df_users_ex = pd.read_csv(TESTING_SET_PATH, sep=",", header=0)
df_users_ex['cls'] = None
df_users = pd.concat([df_users_train, df_users_ex])
df_users.head()

# Part 2 - Compute the distribution of the target variable
counts, bins = np.histogram(df_users_train["cls"], bins=[0,1,2])

# Plot the distribution
pl.figure(figsize=(6,6))
pl.bar(bins[:-1], counts, width=0.5, alpha=0.4)
pl.xticks(bins[:-1] + 0.3, ["negative", "positive"])
pl.xlim(bins[0] - 0.5, bins[-1])
pl.ylabel("Number of users")
pl.title("Target variable distribution")
pl.show()

# Part 3 -- Get coords function. Pls, don`t look at secret keys :(
CONSUMER_KEY = "RAhs7ThpKFL14Nb188tdkdLkX"
CONSUMER_SECRET = "V7VaBOgAnOnEk0Q3v9A0MKygAYwJqboNtFfsURIP52Cp2ar2EX"

ACCESS_TOKEN_KEY = "430060558-rBilaliqRUoDm9JHTeSsWUpM3BsCoDvtKhtul5On"
ACCESS_TOKEN_SECRET = "iqHidUrd9tstB97CWx6fw1xg7gbeAd4QibrFcL9d5JflA"

api = twitter.Api(consumer_key=CONSUMER_KEY,
                  consumer_secret=CONSUMER_SECRET,
                  access_token_key=ACCESS_TOKEN_KEY,
                  access_token_secret=ACCESS_TOKEN_SECRET)

#http://api.geonames.org/search?q="Russia"&maxRows=1&username=GimmeDanger

GEO_USER_NAME = 'GimmeDanger'

coordinates_by_location = {}

def get_coordinates_by_location(location):
    global coordinates_by_location

    if not location in coordinates_by_location.keys():
        url = "http://api.geonames.org/search"
        params = {'q' : location, 'maxRows' : 1, 'username' : GEO_USER_NAME}
        request = requests.get(url, params = params)

        try:
            root = ET.fromstring(request.text)
            geoname = root.find('geoname')
            coordinates_by_location[location] = (geoname.find('lat').text,
                                                 geoname.find('lng').text,
                                                 geoname.find('countryName').text)
        except:
            coordinates_by_location[location] = None

    return coordinates_by_location[location]

#print get_coordinates_by_location("San Francisco, CA")
#print get_coordinates_by_location("Moscow")
#print coordinates_by_location

# Part 4 -- Upload users data
ts_parser = lambda date_str: dateutil.parser.parse(date_str) if pd.notnull(date_str) else None

user_records_file = []
tmp_file_name = 'tmp_user_records'
if os.path.exists(tmp_file_name):
    with open(tmp_file_name) as f:
        for line in f:
            try:
                user_records_file.append(json.loads(line))
            except:
                print "Exeption while reading."
                continue

processed_users = set()
for r in user_records_file:
    processed_users.add(r["uid"])

f = open(tmp_file_name, 'a')


def twitter_user_to_dataframe_record(user):
    dt = ts_parser(user.created_at)
    record = {
        "uid": user.id,
        "name": user.name,
        "screen_name": user.screen_name,
        "created_at": dt.strftime("%Y-%m") if dt else dt,
        "followers_count": user.followers_count,
        "friends_count": user.friends_count,
        "statuses_count": user.statuses_count,
        "favourites_count": user.favourites_count,
        "listed_count": user.listed_count,
        "verified": user.verified
    }

    if user.description is not None and user.description.strip() != "":
        record["description"] = user.description

    if user.location is not None and user.location.strip() != "":
        record["location"] = user.location
        coords = get_coordinates_by_location(user.location)
        if coords is not None:
            record["lat"], record["lon"], record["country"] = coords

    return record

def get_user_records(df):
    # max size of possible q_list len as the argument of api.UsersLookup
    MAX_QUERY_SIZE = 100

    global f, processed_users

    list = df['uid'].tolist()
    users_to_process = len(list)
    q_list = []
    record_list = []

    for uid in list:
        if not uid in processed_users:
            if len(q_list) < MAX_QUERY_SIZE:
                q_list.append(uid)
                users_to_process -= 1
            if len(q_list) == MAX_QUERY_SIZE or users_to_process == 0:
                try:
                    users = api.UsersLookup(q_list)
                    for u in users:
                        record = twitter_user_to_dataframe_record(u)
                        json.dump(record, f)
                        f.write('\n')
                        record_list.append(record)
                except:
                    # Loose a little
                    '''
                    for u in q_list:
                        try:
                            user = api.GetUser(u)
                            record = twitter_user_to_dataframe_record(user)
                            json.dump(record, f)
                            f.write('\n')
                            record_list.append(record)
                        except:
                            print "User is not found"
                    '''
                    pass

                del q_list[:]
    return record_list

user_records = user_records_file + get_user_records(df_users)

# Maximum number of user IDs returned by Twitter's user lookup

f.close()

print "Creating data frame from loaded data"
df_records = pd.DataFrame(user_records,
                          columns=["uid", "name", "screen_name", "description", "verified", "location", "lat", "lon",
                                   "country", "created_at", "followers_count", "friends_count", "statuses_count",
                                   "favourites_count", "listed_count"])
print "Merging data frame with the training set"
df_full = pd.merge(df_users, df_records, on="uid", how="left")
print "Finished building data frame"

# Part 5 -- bar diagram for 'negative' and 'positive' users
def count_users(grouped):
    """
    Counts number of positive and negative users
    created at each date.

    Returns:
        count_pos -- 1D numpy array with the counts of positive users created at each date
        count_neg -- 1D numpy array with the counts of negative users created at each date
        dts -- a list of date strings, e.g. ['2014-10', '2014-11', ...]
    """
    dts = []
    count_pos, count_neg = np.zeros(len(grouped)), np.zeros(len(grouped))
    i = 0
    for k, g in grouped:
        dts.append(k)
        neg, pos = 0, 0
        for x in g['cls']:
            if x == 1:
                pos += 1
            elif x == 0:
                neg += 1
        count_pos[i] = pos
        count_neg[i] = neg
        i += 1

    return count_pos, count_neg, dts


grouped = df_full.groupby(map(lambda dt: dt if pd.notnull(dt) else "NA", df_full["created_at"]))

count_pos, count_neg, dts = count_users(grouped)

fraction_pos = count_pos / (count_pos + count_neg + 1e-10)
fraction_neg = 1 - fraction_pos

sort_ind = np.argsort(dts)

pl.figure(figsize=(20, 3))
pl.bar(np.arange(len(dts)), fraction_pos[sort_ind], width=1.0, color='red', alpha=0.6, linewidth=0, label="Positive")
pl.bar(np.arange(len(dts)), fraction_neg[sort_ind], bottom=fraction_pos[sort_ind], width=1.0, color='green', alpha=0.6,
       linewidth=0, label="Negative")
pl.xticks(np.arange(len(dts)) + 0.4, sorted(dts), rotation=90)
pl.title("Class distribution by account creation month")
pl.xlim(0, len(dts))
pl.legend()
pl.show()


# Part 7 -- location distibution
pl.figure(figsize=(25,17))

m = bm.Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')

m.drawcountries(linewidth=0.2)
m.fillcontinents(color='lavender', lake_color='#000040')
m.drawmapboundary(linewidth=0.2, fill_color='#000040')
m.drawparallels(np.arange(-90,90,30),labels=[0,0,0,0], color='white', linewidth=0.5)
m.drawmeridians(np.arange(0,360,30),labels=[0,0,0,0], color='white', linewidth=0.5)

#http://matplotlib.org/basemap/users/examples.html

def plot_points_on_map(df_full):
    """
    Plot points on the map.
    """
    LON = 0
    LAT = 1
    NUM = 2
    CLS = 3

    dct = {}
    row_len = len(df_full['uid'])
    for col in xrange(row_len):
        loc = df_full['location'][col]
        cls = df_full['cls'][col]

        if cls != 1 and cls != 0:
            # skip NANs
            continue

        if loc in dct:
            dct[loc][NUM] += 1
            if cls > 0:
                dct[loc][CLS] += 1
            else:
                dct[loc][CLS] -= 1
        else:
            lat = df_full['lat'][col]
            lon = df_full['lon'][col]
            signed_cls = df_full['cls'][col]
            if signed_cls == 0:
                signed_cls = -1
            dct[loc] = [lon, lat, 1, signed_cls]

    for k, v in dct.iteritems():
        x = v[LON]
        y = v[LAT]
        r = v[NUM]
        c = v[CLS]
        usr_color = 'red'
        if c < 0:
            usr_color = 'green'
        m.scatter(x, y, max (r, 20) , marker='o', zorder=200, color=usr_color)

    # labels
    m.scatter(0, 0, 10, marker='o', zorder=200, label='Positive', color='red')
    m.scatter(0, 0, 10, marker='o', zorder=200, label='Negative', color='green')
    return

plot_points_on_map(df_full)

pl.title("Geospatial distribution of twitter users")
pl.legend()
pl.show()

#Part 8 -- stat and hist
sample_number = 500
users_wth_neg_class = df_full[df_full["cls"]==0].sample(sample_number)
users_wth_pos_class = df_full[df_full["cls"]==1].sample(sample_number)

def descriptive_stat_and_hist(users_wth_neg_class, users_wth_pos_class):
    print '        median, mean, max'
    print 'Negative: %d %d %d' % (users_wth_neg_class["followers_count"].median(),
                                  users_wth_neg_class["followers_count"].mean(),
                                  users_wth_neg_class["followers_count"].max())
    print 'Positive: %d %d %d' % (users_wth_pos_class["followers_count"].median(),
                                  users_wth_pos_class["followers_count"].mean(),
                                  users_wth_pos_class["followers_count"].max())

    pl.figure(figsize=(6, 6))
    pl.hist([users_wth_neg_class["followers_count"], users_wth_pos_class["followers_count"]],
            fill=True, range=(0, 25000), color=['green', 'red'], label=['Negative', 'Positive'])

    pl.xlabel('Value')
    pl.ylabel('Frequency')
    pl.title('Followers number')
    pl.legend()
    pl.show()

    return


sample_number = 500
users_wth_pos_class = df_full[df_full["cls"] == 1].sample(sample_number)
users_wth_neg_class = df_full[df_full["cls"] == 0].sample(sample_number)

descriptive_stat_and_hist(users_wth_neg_class, users_wth_pos_class)

# Part 9 -- save data
OUT_FILE_PATH = "hw1_out.csv"
print "Saving output data frame to %s" % OUT_FILE_PATH
df_full.to_csv(OUT_FILE_PATH, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
df_full.head()
