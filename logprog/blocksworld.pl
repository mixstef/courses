% blockword problem - basic planner

% ------planning space--------

% objects in blocksworld - blocks and places
% blocks can be moved and stacked on top of a place or another block
block(a).
block(b).
block(c).
% places can be used only to stack blocks on
place(1).
place(2).
place(3).
place(4).
% a collective definition of an object: a block or a place
object(X) :- place(X).
object(X) :- block(X).

% precondition can(Action,Cond) - all these should be true for the move to be possible
can(move(Block,From,To),[clear(Block),clear(To),on(Block,From)]) :-
	block(Block),
	object(To),
	\+(To = Block),
	object(From),
	\+(From = To),
	\+(Block = From).

% effects after the move action
% these are true after move
adds(move(Block,From,To),[on(Block,To),clear(From)]).
% these are false after move
deletes(move(Block,From,To),[on(Block,From),clear(To)]).

% -------main planner-----------
% the planner defs: plan(State,Goals,Plan,FinalState)

% the empty plan - we are already in initial state
plan(State,Goals,[],State) :- satisfied(State,Goals).

plan(State,Goals,Plan,FinalState) :-
        concat(PrePlan,[Action|PostPlan],Plan),         % divide Plan
        selectgoal(State,Goals,Goal),                   % select a Goal
        achieves(Action,Goal),                          % relevant Action
        can(Action,Condition),                          % pre-cond of Action
        plan(State,Condition,PrePlan,MidState1),        % enable Action
        apply(MidState1,Action,MidState2),              % apply Action
        plan(MidState2,Goals,PostPlan,FinalState).      % do remaining goals


% -------helper functions-----------

% satisfied(State,Goals): Goals are true in State
satisfied(_,[]).
satisfied(State,[Goal|Goals]) :- member(Goal,State),satisfied(State,Goals).

% select a Goal from a list of Goals that is not already in State
selectgoal(State,Goals,Goal) :- member(Goal,Goals),\+(member(Goal,State)).

% achieves(Action,Goal) : Goal is a relationship added by Action
achieves(Action,Goal) :- adds(Action,Goals),member(Goal,Goals).

% apply(State,Action,NewState) : when Action from State produces NewState
apply(State,Action,NewState) :-
        deletes(Action,DelList),
        listdiff(State,DelList,State1),!,
        adds(Action,AddList),
        concat(AddList,State1,NewState).
        
% --------misc list operations--------

% concat(L1,L2,L3): L3 is the concatenation of L1 and L2
concat([],L,L).
concat([X|L1],L2,[X|L3]) :- concat(L1,L2,L3).

% listdiff(L1,L2,Diff): Diff is the set diff L1-L2
listdiff([],_,[]).
listdiff([X|L1],L2,Diff) :- member(X,L2),!,listdiff(L1,L2,Diff).
listdiff([X|L1],L2,[X|Diff]) :- listdiff(L1,L2,Diff).


