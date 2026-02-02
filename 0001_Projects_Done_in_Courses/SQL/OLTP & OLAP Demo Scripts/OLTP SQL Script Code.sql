create database finance_shares;
use finance_shares;

create table customers(
customers_id int primary key,
name varchar(20) not null,
address varchar(20));

create table shares(
script_code int primary key,
customers_id int,
name varchar(20) not null,
high int,
low int,
foreign key (customers_id) references customers(customers_id));

create table profits(
customers_id int,
script_code int,
profits int,
foreign key (customers_id) references customers(customers_id),
foreign key (script_code) references shares(script_code));

insert into customers (customers_id,name,address) 
values (1,"Anuj","Noida"),(2,"Guatam","Inderapuram");

insert into shares (script_code,customers_id,name,high,low)
values (1050,1,"reliance",200,100), (1051,2,"Tata",205,305);

insert into profits values
(1,1050,20000),
(1,1051,15360),
(2,1050,36060);


