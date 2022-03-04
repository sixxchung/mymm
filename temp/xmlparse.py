# XML (eXtensible Markup Language
# elements
# <tag attrib="attribute"> text </tag>

import xml.etree.ElementTree as ET

### Parsing
tree = ET.parse('./text.xml')
root = tree.getroot()
print(f'tag=>{root.tag}  attrib=>{root.attrib} text=>{root.text}')

### parse with loop
for child in root:
    print(f'tag=>{child.tag}  attrib=>{child.attrib} text=>{child.text}')

[elem.tag    for elem in root.iter()]
[elem.attrib for elem in root.iter()]
[elem.text   for elem in root.iter()]

for rating in root.iter('rating'):
    print(f'tag=>{rating.tag}  attrib=>{rating.attrib} text=>{rating.text}')

### Scan with XPath
#- Xpath with tag
for movie in root.findall("./genre/decade/movie/[year='2000']"):
    print(f'tag=>{movie.tag}  attrib=>{movie.attrib} text=>{movie.text}')
  
#- Xtpah with attrib
for movie in root.findall("./genre/decade/movie/format/[@multiple='Yes']"):
    print(f'tag=>{movie.tag}  attrib=>{movie.attrib} text=>{movie.text}')
# ... parent element
for movie in root.findall("./genre/decade/movie/format/[@multiple='Yes']..."):
    print(f'tag=>{movie.tag}  attrib=>{movie.attrib} text=>{movie.text}')

#### Modify and Save 
b2tf = root.find("./genre/decade/movie[@title='Back 2 the Future']")
print(b2tf)
b2tf.attrib["title"] = 'Back Back Back'
tree.write("movies.xml")

# search with regex
import re
for form in root.findall("./genre/decade/movie/format"):
    print(re.search(',', form.text))

# add (modify) attribute with set()
import re
for form in root.findall("./genre/decade/movie/format"):
    match = re.search(',', form.text)
    if match:
        form.set('multiple', 'yes')
    else:
        form.set('multiple', 'No') 

for form in root.findall("./genre/decade/movie/format"):
    print(form.attrib, form.text)

# find wrong data
for decade in root.findall("./genre/decade"):
    print(decade.attrib)
    for year in decade.findall("./movie/year"):
        print(year.text)

for movie in root.findall("./genre/decade/movie/[year='2000']"):
    print(movie.attrib)

action = root.find("./genre[@category='Action']")
new_dec = ET.SubElement(action, 'decade')
new_dec.attrib["years"] = '2000s'

xmen = root.find("./genre/decade/movie[@title='X-Men']")
dec2000s = root.find("./genre[@category='Action']/decade[@years='2000s']")
dec2000s.append(xmen)
dec1990s = root.find("./genre[@category='Action']/decade[@years='1990s']")
dec1990s.remove(xmen)

# build the new xml
tree.write("movies.xml")
tree = ET.parse('movies.xml')
root = tree.getroot()
print(ET.tostring(root, encoding='utf8').decode('utf8'))