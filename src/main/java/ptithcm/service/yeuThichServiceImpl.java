package ptithcm.service;

import java.util.List;

import javax.transaction.Transactional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import ptithcm.dao.YeuThichDAO;
import ptithcm.entity.YeuThichEntity;

@Service
@Transactional
public class yeuThichServiceImpl implements yeuThichService{
	@Autowired
	YeuThichDAO yeuThichDao;

	@Override
	public List<YeuThichEntity> layDSYeuThichCuaUser(int maNd) {
		// TODO Auto-generated method stub
		return yeuThichDao.layDSYeuThichCuaUser(maNd);
	}

	@Override
	public YeuThichEntity layYeuThichTheoMaNDVaSanPham(int maNd, String maSp) {
		// TODO Auto-generated method stub
		return yeuThichDao.layYeuThichTheoMaNDVaSanPham(maNd, maSp);
	}

	@Override
	public void addYeuThich(YeuThichEntity yeuThich) {
		// TODO Auto-generated method stub
		yeuThichDao.addYeuThich(yeuThich);
	}

	@Override
	public void deleteYeuThich(int maYT) {
		// TODO Auto-generated method stub
		yeuThichDao.deleteYeuThich(maYT);
	}
	
}
