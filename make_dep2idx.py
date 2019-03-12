# deps = "acl	acomp advcl	advmod agent amod appos	attr aux auxpass case cc ccomp compound conj cop csubj csubjpass dative dep	det	dobj expl intj mark meta neg nn nounmod	npmod nsubj nsubjpass nummod oprd obj obl parataxis pcomp pobj poss preconj prep prt punct quantmod	relcl root xcomp"
import json

def main():
	# print(deps)
	# deps2 = deps.split(" ")
	# print(len(deps2))

	deps = "acl acomp advcl advmod agent amod appos attr aux auxpass case cc ccomp compound conj cop csubj csubjpass dative dep det dobj expl intj mark meta neg nn nmod npmod nsubj nsubjpass nummod oprd obj obl parataxis pcomp pobj poss preconj prep prt punct quantmod relcl root xcomp"
	deps = deps.split(" ")
	print(len(deps))
	dep2idx = {}
	for i in range(len(deps)):
		dep2idx[deps[i].upper()] = i
		dep2idx[deps[i]] = i
	print(dep2idx)

	with open('dep2idx.json', 'w') as outfile:
		json.dump(dep2idx, outfile)





if __name__ == '__main__':
    main()