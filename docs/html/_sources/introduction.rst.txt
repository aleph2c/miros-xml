
  *Decisions without actions are pointless. Actions without decisions are reckless.* 
  
  -- John Boyd

.. toctree::
   :titlesonly:

.. _introduction-introduction:

Introduction
============

Problem - The Tale of Two Architectural Explosion Chambers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Software engineers like to describe how their systems should behave with a
picture. Often these pictures look like a bunch of circles connected by arrows.
The circles represent the different system states and the arrows represent the
things which cause change.  Pictures drawn like this are called finite state
machines (FSMs).

The problem with FSMs are that they are architectural time bombs.  As the number
of requirements increase in your system, the number of bubbles and arrows
required to model it, outpaces the number of requirements.  This
effect is so dramatic, it has been given the name: "state-space-explosion".

Any useful engineering formalism (or way of drawing your problem) should
compress complexity within the model, not make your model more complex than the
real system.

.. image:: _static/bomb_disposal.jpg
    :target: https://www.ocregister.com/2008/08/04/officials-explosive-could-have-blasted-4-homes/
    :class: noscale-center

In 1987, David Harel neatly solved "state space explosions" by
inventing some new drawing rules for these circle-and-arrow type of pictures.
He named these drawing rules: "hierarchical state machines" (HSMs) and "parallel
regions".  A hierarchical state machine allows for circles to be drawn within
circles; an inner circle containing all of the behavior described by an outer
circle.  A parallel region, is represented as a dashed line on a picture.  Each
region marked off with this dashed line describes concurrent operation.

David Harel's drawing rules ensnare tremendous amounts of behavioral complexity.
If you model systems using his techniques you will end up with small and
intuitive diagrams which can easily adapt to future requirements.

When state machines are running in parallel regions they are running at the same
time: both regions react to the events which are sent to the statechart.  This
means that your statemachine can have more than one active state.  Such an idea
is easy for a theorist to envision, but it is much harder for the practitioner
to implement.  Regions can exist within regions, and each region represents a
parallel execution of code, with access to common outer memory and a need to
react to events managed by other parts of the program.  It is like applying
topology to your system forks.

To make the manifestation of such pictures easier for a software engineer, firms
wrote tools to help draw these pictures then turn them into working software.
The problem was that the tooling, though powerful, was expensive, added risk and
produced programs outside of the direct control of the software developer.
To get your code working, you had to ensure a third party tool was installed,
could run properly, that its licence server was working and linked in your
network, and so on.  The word for this is Vender-lock-in.  If the firm you
depended on failed, your entire software system could become moot.

Miro Samek liberated statecharts from Vender-lock-in with his
2000 paper titled `State-Oriented Programming
<https://www.embedded.com/state-oriented-programming/>`_.  His algorithm
supported the bubble within bubble part of the statemachine picture, but it did
not support the dashed line: parallel regions. He went about addressing the
"state space explosion" problem in a different way, he invented the `orthogonal
component pattern
<https://aleph2c.github.io/miros/patterns.html#patterns-orthogonal-component>`_:
an HSM running within another HSM.

Miro Samek's approach was built for small processors, and it ran fast and used
very few resources.  It was not as pictorially expressive as the parallel
region, but it was comprehensible by the software developer who was using it.
To use his software, an engineer would port it onto their processor, just as
they would with a real-time operating system (RTOS).  In fact, with his software
you didn't need an RTOS anymore.

.. note::

  `Miro Samek's business model was interesting too
  <https://www.state-machine.com/>`_.  He provided his code and his
  understanding for free under an open source licence, and had another licence
  for commercial products.  For this reason there is no Vender-lock-in, you can
  just go an get his code and build it into your system directly.  His business
  innovation was based on trust:  Engineers working within large firms want to
  see that his firm gets paid, so they will research the licences on his webpage
  and make sure he gets his money.  In doing so they establish a relationship
  which could save them months in development time.

Both David Harel and Miro Samek went about containing the architecture "state
space explosion" caused from FSMs.  One method was practical from an
implementation perspective and the other was more pictorially expressive in how
it described concurrency.

This project is about having cake and eating it too.  The library this project
is dependent upon uses Miro Samek's algorithm and thereby doesn't implicitly
support parallel regions, but it does support the orthogonal component pattern.
So I will use the orthogonal component as a kind of assembly language.  I will
map a parallel region described within an XML document (within a <p> tag) onto a
set of orthogonal regions.  One orthogonal region may contain 0 or more other
orthogonal regions, in a kind of hierarchy of hierarchical state machines
(HHSM).  An event will be able to cause transitions between these orthogonal
regions. From the outside, it will look like parallel region's are supported.
So this library is about bringing parallel regions to the Miro Samek algorithm.

The drawing rule to make these HHSMs will just be those described by David
Harel, and yet the code will follow the Miro Samek philosophy of being
comprehensible and accessible.

.. _introduction-what-this-documentation-will-provide:


