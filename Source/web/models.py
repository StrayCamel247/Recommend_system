from django.contrib.auth.models import Permission, User
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

class Novel(models.Model):
	title   	= models.CharField(max_length=200)
	genre  		= models.CharField(max_length=100)
	movie_logo  = models.FileField()

	def __str__(self):
		return self.title

class Myrating(models.Model):
	user   	= models.ForeignKey(User,on_delete=models.CASCADE) 
	novel 	= models.ForeignKey(Novel,on_delete=models.CASCADE)
	rating 	= models.IntegerField(default=1,validators=[MaxValueValidator(5),MinValueValidator(0)])
		