

def hasMissingWeeks(weeks):
	for i in range(min(weeks), max(weeks)):
		if i not in weeks:
			return True
	return False
