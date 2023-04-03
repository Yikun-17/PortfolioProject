
--select *
--from CovidVacinations$
--order by 3,4

--looking at total cases vs total deaths
--shows likelihood of dying if you contract covid in your country

select Location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
from CovidDeaths$
where location like '%states%'
order by 1,2

--looking at total cases vs population
--shows what percentage of population got covid

select Location, date, population, total_cases, (total_cases/population)*100 as Percentofpopulationinfected
from CovidDeaths$
--where location like '%states%'
order by 1,2

--looking at countries with higest infection rate compared to population

select Location, population, MAX(total_cases) as HighestInfectionCount, MAX((total_cases/population))*100 as PercentPopulationInfected
from CovidDeaths$
--where location like '%states%'
group by Location,population
order by PercentPopulationInfected desc

--showing countries with highest death count per population

select Location, max(cast(total_deaths as int)) as totaldeathcount
from CovidDeaths$
--where location like '%states%'
where continent is not null
group by Location
order by totaldeathcount desc


--let's break things down by continent

select continent, max(cast(total_deaths as int)) as totaldeathcount
from CovidDeaths$
--where location like '%states%'
where continent is not null
group by continent
order by totaldeathcount desc

--showing continents with highest death counts

select continent, max(cast(total_deaths as int)) as totaldeathcount
from CovidDeaths$
--where location like '%states%'
where continent is not null
group by continent
order by totaldeathcount desc

--global numbers 

select sum(new_cases) as total_cases, sum(cast(new_deaths as int)) as total_deaths, sum(cast(new_deaths as int))/sum(new_cases)*100 as deathpercentage
from CovidDeaths$
--where location like '%states%'
where continent is not null
--group by date
order by 1,2

--looking for total population vs vaccinations

select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,sum(convert(int,vac.new_vaccinations)) over(partition by dea.location order by dea.location, dea.date) as rollingpeoplevaccinated, (rollingpeoplevaccinated/population)*100
from CovidDeaths$ dea
join CovidVacinations$ vac
	on dea.location= vac.location
	and dea.date=vac.date
where dea.continent is not null
order by 2,3

-- use cte

with popvsvac (continent,location,date,population, new_vaccinations, rollingpeoplevaccinated)
as
(
select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
sum(convert(int,vac.new_vaccinations)) over(partition by dea.location order by dea.location, 
dea.date) as rollingpeoplevaccinated--, (rollingpeoplevaccinated/population)*100
from CovidDeaths$ dea
join CovidVacinations$ vac
	on dea.location= vac.location
	and dea.date=vac.date
where dea.continent is not null
--order by 2,3
)
select*,(rollingpeoplevaccinated/population)*100
from popvsvac


--temp table

drop table if exists #percentpopulationvaccinated
create table #percentpopulationvaccinated
(
continent nvarchar(255),
location nvarchar(255),
date datetime,
population numeric,
new_vaccinations numeric,
rollingpeoplevaccinated numeric
)

insert into #percentpopulationvaccinated
select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
sum(convert(int,vac.new_vaccinations)) over(partition by dea.location order by dea.location, 
dea.date) as rollingpeoplevaccinated--, (rollingpeoplevaccinated/population)*100
from CovidDeaths$ dea
join CovidVacinations$ vac
	on dea.location= vac.location
	and dea.date=vac.date
--where dea.continent is not null
--order by 2,3

select *, (rollingpeoplevaccinated/population)*100
from #percentpopulationvaccinated

--creating view to store data for later visualizations

create view percentpopulatiovaccinated as
select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
sum(convert(int,vac.new_vaccinations)) over(partition by dea.location order by dea.location, 
dea.date) as rollingpeoplevaccinated--, (rollingpeoplevaccinated/population)*100
from CovidDeaths$ dea
join CovidVacinations$ vac
	on dea.location= vac.location
	and dea.date=vac.date
where dea.continent is not null
--order by 2,3

select*
from percentpopulatiovaccinated



