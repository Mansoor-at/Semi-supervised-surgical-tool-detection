import json

labels= {"Grasper":1, "Bipolar":2, "Hook":3, "Scissors":4, "Clipper":5, "Irrigator":6, "SpecimenBag":7}

#json_object = json.dumps(labels, indent = 4)

with open('sample.json', 'w') as f:
    json.dump(labels, f)