# decision: charmq integration

Prerequisite: knshnb B6/B7 ensemble has a valid submission and recorded score.

Use charmq only if at least one of these is true:

- B6+B7 score is below target and there are at least three full days left.
- B7 failed and charmq B7 or another charmq model looks faster to adapt.
- Existing charmq artifacts can be converted into knshnb ensemble format with a small adapter.

Do not merge charmq training code into knshnb. Integration target is prediction artifacts consumed by knshnb ensemble.

If charmq starts, record:

- environment command
- dependency versions
- model selected
- bbox variants inferred
- output schema
- adapter changes, if any
